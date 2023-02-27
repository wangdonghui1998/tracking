import torch.nn as nn
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


#单向lstm---cha
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0)) # output(5, 30, 64)
        pred = self.linear(output)  # (25, 5, 2)
        pred = pred[:, -1, :]  # (25,2)
        return pred

#双向LSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2  # 双向lstm
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)  # [6，25，64]
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)  # [6，25，64]
        input_seq = input_seq.view(batch_size, seq_len, self.input_size)  # [25,5,2]
        # (batch_size, seq_len, 2 * hidden_size)包含两个方向的输出，output[0]为序列从左往右第一个隐藏层状态输出和序列从右往左最后一个隐藏层状态输出的拼接；output[-1]为序列从左往右最后一个隐藏层状态输出和序列从右往左第一个隐藏层状态输出的拼接。
        output, _ = self.lstm(input_seq, (h_0, c_0))  # [25,5,128]
        output = output.contiguous().view(batch_size, seq_len, self.num_directions, self.hidden_size)  # [25,5,2,64]
        output = torch.mean(output, dim=2)  # [25,5,64]
        pred = self.linear(output)  # pred()  [25,5,2]
        pred = pred[:, -1, :]  # [25,2]
        return pred

class CNN_LSTM(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size,hidden_size, num_layers, output_size):
        super(CNN_LSTM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=30, seq_len=24, input_size=7) ---> permute(0, 2, 1)
        # (30, 7, 24)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        # (batch_size=30, out_channels=32, seq_len-4=20) ---> permute(0, 2, 1)
        # (30, 20, 32)
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x[:, -1, :]
        return x

class CNN_LSTM2(nn.Module):
    def __init__(self, in_channels,out_channels, kernel_size,hidden_size, num_layers, output_size):
        super(CNN_LSTM2, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # (batch_size=30, seq_len=24, input_size=7) ---> permute(0, 2, 1)
        # (30, 7, 24)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        # (batch_size=30, out_channels=32, seq_len-4=20) ---> permute(0, 2, 1)
        # (30, 20, 32)
        self.lstm1 = nn.LSTM(input_size=out_channels, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,dropout=0.2)
        self.lstm2 = nn.LSTM(input_size=out_channels*2, hidden_size=hidden_size*2,
                             num_layers=num_layers, batch_first=True,dropout=0.2)

        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x)
        x = x[:, -1, :]

        return x
