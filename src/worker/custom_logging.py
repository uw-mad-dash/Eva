import datetime
import logging

class WorkerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        worker = self.extra.get("worker")
        timestamp = worker.get_current_timestamp()
        # timestamp to 2 decimal places
        timestamp = "{:.2f}".format(timestamp)
        # time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return '[%s] %s' % (timestamp, msg), kwargs