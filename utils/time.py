import time


class time_calculator():

    def __init__(self):
        self.start_time = time.time()

    def time_length(self):

        time_period = time.time() - self.start_time

        hour = int(time_period // 3600)
        minute = int((time_period - 3600 * hour) // 60)
        second = time_period - 3600 * hour - 60 * minute

        return f'{hour}H {minute}M {second:.2f}S'
