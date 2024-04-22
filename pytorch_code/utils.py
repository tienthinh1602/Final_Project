import networkx as nx
import numpy as np


def build_graph(train_data):
    graph = nx.DiGraph()
    # Duyệt qua mỗi chuỗi trong dữ liệu huấn luyện
    for seq in train_data:
        # Duyệt qua từng phần tử trong chuỗi
        for i in range(len(seq) - 1):
            # Xác định trọng số của cạnh
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            # Thêm cạnh vào đồ thị với trọng số đã xác định
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    # Tính trọng số chuẩn hóa cho các cạnh vào mỗi đỉnh
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    # Độ dài của mỗi chuỗi người dùng
    us_lens = [len(upois) for upois in all_usr_pois]
    # Độ dài lớn nhất trong tất cả các chuỗi
    len_max = max(us_lens)
    # Thêm item_tail vào cuối mỗi chuỗi
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    # Tạo ma trận mask, giữ 1 cho phần tử thực tế, 0 cho phần tử thêm vào
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    # Giải nén tập huấn luyện thành features và labels
    train_set_x, train_set_y = train_set 
     # Số lượng mẫu trong tập huấn luyện
    n_samples = len(train_set_x)
    # Tạo một mảng các chỉ số sắp xếp tăng dần
    sidx = np.arange(n_samples, dtype='int32')
    # Xáo trộn mảng chỉ số
    np.random.shuffle(sidx)

    # Tính số lượng mẫu cho tập kiểm tra dựa trên valid_portion
    n_train = int(np.round(n_samples * (1. - valid_portion)))

    # Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None, opt=None):
        # Xử lý dữ liệu đầu vào
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        # Chuyển đổi thành mảng NumPy
        self.inputs = np.asarray(inputs)
        # Chuyển đổi thành mảng NumPy
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        # Nếu opt.dynamic được bật, 
        # sử dụng data[2] thay vì data[1] cho self.targets
        if opt.dynamic:
            self.targets = np.asarray(data[2])
        # Số lượng mẫu dữ liệu
        self.length = len(inputs)
         # Tùy chọn để xáo trộn dữ liệu
        self.shuffle = shuffle
        # Đồ thị (graph) có thể được sử dụng cho mô hình (tùy chọn)
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            # Xáo trộn dữ liệu nếu được yêu cầu
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        # Tính toán số lượng batch
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        # Chia thành các slices (mảng các chỉ số) tương ứng với các batch
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        # Lấy các dữ liệu tương ứng với slice i
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        items, n_node, A, alias_inputs = [], [], [], []

        # Xử lý mỗi chuỗi trong slice
        for u_input in inputs:
            n_node.append(len(np.unique(u_input)))

        max_n_node = np.max(n_node)

        for u_input in inputs:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))

            # Xây dựng ma trận kề từ chuỗi người dùng
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
                
            # Chuẩn hóa ma trận kề
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        
        return alias_inputs, A, items, mask, targets
