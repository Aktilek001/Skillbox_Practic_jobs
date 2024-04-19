import dill
from apscheduler.schedulers.blocking import BlockingScheduler
import tzlocal
import pandas as pd

sched = BlockingScheduler(timezone=tzlocal.get_localzone_name())

df = pd.read_csv('model/data/homework.csv')
with open('model/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)

@sched.scheduled_job('cron', second='*/5')
def on_time():
    data = df.sample(n=5)
    data['preds'] = model['model'].predict(data)
    print(data[['id', 'preds']])


if __name__ == '__main__':
    sched.start()
