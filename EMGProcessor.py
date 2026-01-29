import queue
import threading
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.signal
import pytrigno
from typing import List, Tuple


class EMGProcessor:
    def __init__(self, channel_num=2, sample_fre=10, start_time=None, save_time=None, save=False, save_folder=None):
        self.emg_ip = '192.168.10.10'
        self.emg_channel_num = channel_num
        self.emg_sample_rate = 2000
        self.emg_samples_per_read = int(self.emg_sample_rate/sample_fre)
        self.save_folder = save_folder

        self.plot_flag = True
        self.read_emg_flag = True
        self.save_flag = save

        self.all_emg_data = [[] for _ in range(self.emg_channel_num)]
        self.emg_data_raw = [[] for _ in range(self.emg_channel_num)]
        self.emg_data_smooth = [[] for _ in range(self.emg_channel_num)]
        self.plot_emg_data = []
        self.plot_time_data = []
        self.time_data = []

        self.save_time = save_time
        self.callback_time_step = 0.02
        self.plot_time_step = 0.05
        self.plot_emg_num = 10 * 10  # 100个数据点

        self.data_lock = threading.Lock()
        if start_time is None:
            self.tic_start = time.time()
        else:
            self.tic_start = start_time

    def process_one_channel(self, data: np.ndarray, sampling_rate: float, channel_idx: int) -> np.ndarray:
        data_mean = np.mean(data)

        # b, a = scipy.signal.butter(20, sampling_rate, 'high')
        # data = scipy.signal.filtfilt(b, a, data)

        raw = data - data_mean * np.ones_like(data)
        emgfwr = abs(raw)

        numpasses = 3
        lowpass_rate = 6
        Wn = lowpass_rate / (sampling_rate / 2)
        b, a = scipy.signal.butter(numpasses, Wn, 'low')
        EMGLE = scipy.signal.filtfilt(
            b, a, emgfwr,
            padtype='odd',
            padlen=3 * (max(len(b), len(a)) - 1)
        )

        ref_values = {
            0: 407.11,  # 肱二头肌
            1: 467.59,  # 肱三头肌
            2: 250.37,
            3: 468.31,
            4: 468.31,
            5: 468.31,
            6: 468.31,
            7: 468.31,
        }
        ref = ref_values.get(channel_idx, max(EMGLE))

        return EMGLE / ref

    def emg_rectification(self, emg_queue: queue.Queue) -> Tuple[np.ndarray, np.ndarray]:
        emg_data = emg_queue.get()
        emg_queue.task_done()

        data_act = [[] for _ in range(self.emg_channel_num)]
        emg_mean = np.zeros(self.emg_channel_num)

        for i in range(self.emg_channel_num):
            data_act[i] = self.process_one_channel(
                emg_data[:, i],
                self.emg_sample_rate,
                i
            )

        data_act = np.asarray(data_act)
        for i in range(self.emg_channel_num):
            emg_mean[i] = np.clip(data_act[i, :].mean(), 0, 1)

        return emg_data, emg_mean

    def read_emg(self, out_queue: queue.Queue) -> None:
        try:
            dev = pytrigno.TrignoEMG(
                channel_range=(0, 0),
                samples_per_read=self.emg_samples_per_read,
                host=self.emg_ip
            )
            dev.set_channel_range((0, self.emg_channel_num - 1))
            dev.start()

            print(f'{self.emg_channel_num}-channel EMG connected.')
            # if self.tic_start is None:
            #     self.tic_start = time.time()

            while self.read_emg_flag:
                data = dev.read() * 1e6
                assert data.shape == (self.emg_channel_num, self.emg_samples_per_read)
                out_queue.put(np.transpose(np.array(data), (1, 0)))

        # except Exception as e:
        #     print(f"EMG读取错误: {e}")
        finally:
            dev.stop()
            print(f'Stop reading EMG.')

    def process_emg(self, emg_queue: queue.Queue) -> None:
        try:
            while self.read_emg_flag:
                if self.save_time is not None and time.time() - self.tic_start > self.save_time:
                    self.read_emg_flag = False
                if time.time() - self.tic_start > self.callback_time_step:
                    emg, emg_mean = self.emg_rectification(emg_queue)

                    with self.data_lock:
                        for i in range(self.emg_channel_num):
                            self.emg_data_raw[i] = np.append(
                                self.emg_data_raw[i],
                                emg[:, i]
                            )

                        # print(self.emg_data_smooth)
                        # print(self.emg_data_smooth)
                        # print(emg_mean)
                        for i in range(self.emg_channel_num):
                            self.emg_data_smooth[i] = np.append(
                                self.emg_data_smooth[i],
                                emg_mean[i]
                            )[-3:]  # 只保留最后3个值
                        # print(self.emg_data_smooth)

                        emg_data_mean = np.zeros(self.emg_channel_num)
                        for i in range(self.emg_channel_num):
                            emg_data_mean[i] = self.emg_data_smooth[i].mean()
                            self.all_emg_data[i] = np.append(
                                self.all_emg_data[i],
                                emg_data_mean[i]
                            )

                        self.time_data = np.append(
                            self.time_data,
                            time.time() - self.tic_start
                        )

                        if len(self.all_emg_data[0]) > self.plot_emg_num:
                            for i in range(len(self.all_emg_data)):
                                self.plot_emg_data[i] = self.all_emg_data[i][-self.plot_emg_num:]
                            self.plot_time_data = self.time_data[-self.plot_emg_num:]
                        else:
                            self.plot_emg_data = [arr.copy() for arr in self.all_emg_data]
                            self.plot_time_data = self.time_data.copy()

        finally:
            self.plot_flag = False
            self.read_emg_flag = False

            if self.save_flag:
                if self.save_folder:
                    np.save(f'{self.save_folder}/muscle_activation_raw.npy', self.emg_data_raw)
                    np.save(f'{self.save_folder}/muscle_activation_smooth.npy', self.all_emg_data)
                    np.save(f'{self.save_folder}/emg_time.npy', self.time_data)
                    print('EMG date is saved.')
                else:
                    np.save(f'muscle_activation_raw.npy', self.emg_data_raw)
                    np.save(f'muscle_activation_smooth.npy', self.all_emg_data)
                    np.save(f'emg_time.npy', self.time_data)
            print(f'Stop processing EMG.')


    def save_file(self):
        assert self.save_flag is True
        if self.save_folder:
            np.save(f'{self.save_folder}/muscle_activation_raw.npy', self.emg_data_raw)
            np.save(f'{self.save_folder}/muscle_activation_smooth.npy', self.all_emg_data)
            np.save(f'{self.save_folder}/emg_time.npy', self.time_data)
            print('EMG date is saved.')
        else:
            np.save(f'muscle_activation_raw.npy', self.emg_data_raw)
            np.save(f'muscle_activation_smooth.npy', self.all_emg_data)
            np.save(f'emg_time.npy', self.time_data)


    def update_plot(self, frame: int) -> List[plt.Line2D]:
        with self.data_lock:
            time_p = self.plot_time_data.copy()
            emg = self.plot_emg_data.copy()

        if len(time_p) == len(emg[0]) == len(emg[1]):
            y_data = (emg[0] + emg[1]) / 2
            self.line.set_data(time_p, y_data)

            # 动态调整坐标范围
            self.ax.set_xlim(max(0, time_p[-1] - 5), time_p[-1])
            self.ax.set_ylim(0, 1)

        return [self.line]

    def init_plot(self) -> None:
        while len(self.plot_emg_data) == 0:
            time.sleep(0.05)

        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot([], [], lw=2)

        # 设置右侧y轴
        self.ax.yaxis.tick_right()
        self.ax.yaxis.set_label_position("right")
        self.ax.set_ylabel('Muscle Activation')
        self.ax.set_xlabel('Time (s)')
        self.ax.tick_params(labelsize=12)
        self.ax.grid(True, linestyle='--', alpha=0.3)

    def plot_dynamic_image(self) -> None:
        self.init_plot()

        ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            frames=iter(int, 1),
            interval=50,
            blit=False
        )

        plt.show()


def main():
    emg_processor = EMGProcessor(save_time=10)
    data_queue = queue.Queue()
    threads = [
        threading.Thread(
            target=emg_processor.read_emg,
            args=(data_queue,),
            name="EMG-Reader"
        ),
        threading.Thread(
            target=emg_processor.process_emg,
            args=(data_queue,),
            name="EMG-Processor"
        )
    ]

    for t in threads:
        t.daemon = True
        t.start()

    emg_processor.plot_dynamic_image()

    data_queue.join()
    for t in threads:
        t.join()


if __name__ == '__main__':
    main()
