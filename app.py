from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import gzip
import pickle
from datetime import date

app = Flask(__name__)


filepath = "model.rf"
with gzip.open(filepath, 'rb') as f:
    p = pickle.Unpickler(f)
    rf = p.load()

# Functions
column = ['duration', 'days_left', 'airline_Air_India', 'airline_GO_FIRST',
          'airline_Indigo', 'airline_SpiceJet', 'airline_Vistara',
          'source_city_Chennai', 'source_city_Delhi', 'source_city_Hyderabad',
          'source_city_Kolkata', 'source_city_Mumbai',
          'departure_time_Early_Morning', 'departure_time_Evening',
          'departure_time_Late_Night', 'departure_time_Morning',
          'departure_time_Night', 'stops_two_or_more', 'stops_zero',
          'destination_city_Chennai', 'destination_city_Delhi',
          'destination_city_Hyderabad', 'destination_city_Kolkata',
          'destination_city_Mumbai', 'class_Economy']


def class_fun(cl):
    class_dict = {"Economy": "class_Economy",
                  "Business": 0}
    return class_dict.get(cl, -1)


def time_fun(tim):
    time_dict = {
        "Early Morning": "departure_time_Early_Morning",
        "Evening": "departure_time_Evening",
        "Late Night": "departure_time_Late_Night",
        "Morning": "departure_time_Morning",
        "Night": "departure_time_Night",
        "Afternoon": 0
    }

    return time_dict.get(tim, -1)


def duration_fun(dur):
    duration_dict = {"Medium": round(np.random.uniform(2.2, 3.1), 2),
                     "Short": round(np.random.uniform(1.2, 2.2), 2),
                     "Long": round(np.random.uniform(3.1, 4), 2)}
    return duration_dict.get(dur, -1)


def stop_fun(stop):
    stop_dict = {"1": 0,
                 "2+": "stops_two_or_more",
                 "0": 'stops_zero'}
    return stop_dict.get(stop, -1)


def airline_fun(airline):
    airline_dict = {
        "Air India": "airline_Air_India",
        "Go First": "airline_GO_FIRST",
        "Indigo": "airline_Indigo",
        "SpiceJet": "airline_SpiceJet",
        "Vistara": "airline_Vistara",
        "AirAsia": 0

    }

    return airline_dict.get(airline, -1)


def source_fun(source):
    source_dict = {
        "Delhi": "source_city_Delhi",
        "Mumbai": "source_city_Mumbai",
        "Kolkata": "source_city_Kolkata",
        "Chennai": "source_city_Chennai",
        "Hyderabad": "source_city_Hyderabad",
        "Bangalore": 0
    }

    return source_dict.get(source, -1)


def destination_fun(destination):
    destination_dict = {
        "Delhi": "destination_city_Delhi",
        "Mumbai": "destination_city_Mumbai",
        "Kolkata": "destination_city_Kolkata",
        "Chennai": "destination_city_Chennai",
        "Hyderabad": "destination_city_Hyderabad",
        "Bangalore": 0
    }

    return destination_dict.get(destination, -1)


def days_fun(month, day):
    days = int(str(date(2022, int(month), int(day)) - date.today()).split()[0])
    return days


@app.route('/home')
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        source = request.form['from']
        destination = request.form['to']
        month = request.form['month']
        day = request.form['day']
        time = request.form['time']
        airline = request.form['airline']
        duration = request.form['duration']
        stop = request.form['stop']
        cla = request.form['class']

    df_predict = pd.DataFrame([[0]*len(column)], columns=column)

    cl = class_fun(cla)
    tim = time_fun(time)
    dur = duration_fun(duration)
    des = destination_fun(destination)
    sou = source_fun(source)
    sto = stop_fun(stop)
    air = airline_fun(airline)
    days_left = days_fun(month, day)

    if ((cl == -1) | (tim == -1) | (dur == -1) | (des == -1) | (sou == -1) | (sto == -1) | (air == -1)):
        return render_template('index.html', prediction_text="Invalid Input")

    if (destination == source):
        return render_template('index.html', prediction_text="Source and Destination City cannot be the same")

    if (days_left < 0):
        return render_template('index.html', prediction_text="Select Future Date")

    if sto != 0:
        df_predict[sto] = 1
    if cl != 0:
        df_predict[cl] = 1
    if tim != 0:
        df_predict[tim] = 1
    if des != 0:
        df_predict[des] = 1
    if sou != 0:
        df_predict[sou] = 1
    if air != 0:
        df_predict[air] = 1
    df_predict['days_left'] = days_left
    df_predict['duration'] = dur

    pred = round(rf.predict(df_predict)[0], 2)
    pred = round(np.exp(pred), 0)
    return render_template('index.html', prediction_text=f"Price : Rs {pred}")


if __name__ == "__main__":
    app.run(debug=True)
