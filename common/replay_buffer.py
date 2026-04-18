class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.buffer)
    
    def add(self, state, reward, action, next_state, done):
        data = (state, reward, action, next_state, done)
        self.buffer.append(data)

    def get_batch(self):
        # 経験再生がない場合をあえて観察してみる
        states = []
        rewards = []
        actions = []
        next_states = []
        dones = []
        for data in self.buffer:
            states.append(data[0])
            rewards.append(data[1])
            actions.append(data[2])
            next_states.append(data[3])
            dones.append(int(data[4]))
        return torch.Tensor(np.array(states)), torch.Tensor(np.array(rewards)), torch.Tensor(np.array(actions)), torch.Tensor(np.array(next_states)), torch.Tensor(np.array(dones))
