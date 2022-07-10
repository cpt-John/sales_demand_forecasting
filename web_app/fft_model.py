import numpy as np
import pandas as pd


class FFTModel():
    def __init__(self, n=25, limit_n=15, slice_=[None, None, 1]):
        self.n = 25
        self.limit_n = limit_n
        self.slice_ = slice(*slice_)
        self.fit_array = None

    def __filter_fequency(self, fq, Tp=1):
        if fq == 0:
            return False
        fq_d = Tp/fq
        return fq_d > self.limit_n

    def __make_signal(self, fq=1, Ts=1, sf=500, ph=0, amp=1, type=np.sin):
        total_length = 2*np.pi*fq*Ts
        signal = np.linspace(0, total_length, sf, endpoint=True)
        time = np.linspace(0, Ts, sf, endpoint=True)
        signal_wave = amp*type(signal+ph)
        return pd.Series(signal_wave, index=time)

    def fit(self, _X, y):
        fft_out = np.fft.fft(y)
        magnitudes = np.abs(fft_out)[:len(fft_out)//2]

        n = self.n
        frequencies = np.argpartition(magnitudes, -1*n)[-1*n:]

        Tp = len(magnitudes)*2
        def filter_func(fq): return self.__filter_fequency(fq, Tp)
        filter_obj = filter(filter_func, frequencies)
        frequencies = np.array(list(filter_obj))

        complex_form = fft_out[frequencies]
        amp = magnitudes[frequencies]/len(magnitudes)
        ph = np.angle(complex_form)
        data = {"amplitude": amp, "phase": ph}
        signals = pd.DataFrame(data=data, index=frequencies)

        total_fq = pd.DataFrame()
        for fq in signals.index:
            amp, ph = signals.loc[fq][["amplitude", "phase"]].values
            fq_df = self.__make_signal(fq, ph=ph, amp=amp,
                                       sf=len(y), type=np.cos)
            total_fq[f'{fq}'] = fq_df

        sum_frequency = total_fq.sum(axis=1)
        self.fit_array = sum_frequency.iloc[self.slice_].to_numpy()

    def get_fit_array(self, slice_=[None, None, 1]):
        return self.fit_array[slice(*slice_)]

    def predict(self, slice_=[None, None, 1]):
        return self.fit_array[slice(*slice_)]
