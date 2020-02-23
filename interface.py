# -*- coding: utf-8 -*-
"""
@author: jjoachims
"""

# import necessary packages
import pickle, numpy as np, pandas as pd;
import PIL.Image, PIL.ImageTk
import tkinter as tk
from tkinter import ttk
import tkcalendar
import xlrd
from datetime import datetime

# create window
fenster = tk.Tk();

# set window title
fenster.title("Finde DAS Ziel für deinen Bike-Trip!");

# specify path to data
one_hot_path = './models/one_hot_encoder.pkl';
model_path = './models/main_model.pkl';
stations_path = './data/stations.xls';
conversion_path = './data/dict.xlsx';

# create dict for station to id conversion
stationsToId = {};
wb = xlrd.open_workbook(conversion_path);
sh = wb.sheet_by_index(0);   
for i in range(216):
    cell_value_class = sh.cell(i,0).value;
    cell_value_id = sh.cell(i,1).value;
    stationsToId[cell_value_class] = cell_value_id;

# create second dict for id to station conversion
idToStations = {};
wb = xlrd.open_workbook(conversion_path);
sh = wb.sheet_by_index(0);   
for i in range(216):
    cell_value_class = sh.cell(i,1).value;
    cell_value_id = sh.cell(i,0).value;
    idToStations[cell_value_class] = cell_value_id;
    
# create list with station names
stations = [];
wb = xlrd.open_workbook(stations_path);
sh = wb.sheet_by_index(0);
for i in range(216):
    stations.append(sh.cell(i,0).value);

# load one-hot encoder
enc = pickle.load(open(one_hot_path, 'rb'));

# load pretrained model
cclf = pickle.load(open(model_path, 'rb'));
  
def getTop3(day, hour, minute, station_id):
    # normalize features
    minute = round(minute/60, 2);
    hour = round(hour/23., 2);
    day = round(day/6., 2);

    station_id = str(station_id);
    # transform to one-hot vector
    station_id = enc.transform( [[station_id]] ).toarray().tolist();

    # concatenate the features vector
    input = [[day, hour, minute] + station_id[0]];

    # predict probabilities
    res = cclf.predict_proba(input);

    # get the top 3 answers with highest probability
    top3 = res.argsort()[:,-3:][0];

    return [
        str(cclf.classes_[top3[2]]), # 1st most probable
        str(cclf.classes_[top3[1]]), # 2nd most probable
        str(cclf.classes_[top3[0]])  # 3rd most probable
    ];
      
# executed job on button click search destinations
def button_job():
    # get values form the input fields to feed to the model
    startPoint = inputStartpoint.get();
    startPoint = stationsToId[startPoint];
    startHour = inputHour.get();
    startHour = int(startHour);
    startMinute = inputMinutes.get();
    startMinute = int(startMinute);
    startDate = inputDate.get();
    startDate = datetime.strptime(startDate, '%d.%m.%y');
    weekDay = startDate.weekday();  
    
    # get top3 answers from the trained model
    top3 = getTop3(weekDay, startHour, startMinute, startPoint);
    firstStationId = top3[0];
    firstStation = idToStations[firstStationId];
    secondStationId = top3[1];
    secondStation = idToStations[secondStationId];
    thirdStationId = top3[2];
    thirdStation = idToStations[thirdStationId];
    
    # pipe destinations back into the interface
    firstDestination = tk.Label(fenster, text=firstStation, width=35);
    secondDestination = tk.Label(fenster, text=secondStation, width=35);
    thirdDestination = tk.Label(fenster, text=thirdStation, width=35);

    firstDestination.grid(row=5, column=2);
    secondDestination.grid(row=5, column=3);
    thirdDestination.grid(row=5, column=4); 

# adding image1
image_path = './images/bike_Route.png';
image = PIL.Image.open(image_path);

# change size of image1 to fit window
image = image.resize((300, 230), PIL.Image.ANTIALIAS);
photo = PIL.ImageTk.PhotoImage(image);

# adding image2
image2_path = './images/holisticon-logo.png';
image2 = PIL.Image.open(image2_path);

# change size of image2 to fit window
image2 = image2.resize((700, 150), PIL.Image.ANTIALIAS);
photo2 = PIL.ImageTk.PhotoImage(image2);

# create window buttons
findDestinations = tk.Button(fenster, text="Suche Ziele", command=button_job);

# create window labels
firstLabel = tk.Label(fenster, text="1. Ziel:", width=35);
secondLabel = tk.Label(fenster, text="2. Ziel:", width=35);
thirdLabel = tk.Label(fenster, text="3. Ziel:", width=35);
startpointLabel = tk.Label(fenster, text="Gebe deinen Startpunkt ein!");
timeLabel = tk.Label(fenster, text="Gebe eine Uhrzeit ein!");
colonLabel = tk.Label(fenster, text=":");
dateLabel = tk.Label(fenster, text="Gebe ein Datum ein!");
destinationLabel = tk.Label(fenster, text="Hier sind DEINE Ziele:");
photoLabel= tk.Label(fenster, image=photo);
photoLabel.image = image;
photo2Label = tk.Label(fenster, image=photo2);
photo2Label.image = image;

# create input fields
inputStartpoint = ttk.Combobox(fenster, values=stations, state="readonly", width=45);

inputHour = ttk.Combobox(fenster, values=["00","01",
                                          "02","03",
                                          "04","05",
                                          "06","07",
                                          "08","09",
                                          "10","11",
                                          "12","13",
                                          "14","15",
                                          "16","17",
                                          "18","19",
                                          "20","21",
                                          "22","23"], state="readonly");
    
inputMinutes = ttk. Combobox(fenster, values=["00","15","30","45"], state="readonly");

inputDate = tkcalendar.DateEntry(fenster, locale='de_DE', width=12, background='darkblue',
                                 foreground='white', borderwidth=2, year=2020);

# structure window with grid
inputStartpoint.grid(row=1, column=2, pady=30, columnspan=2);
startpointLabel.grid(row=1, column=1);

inputHour.grid(row=2, column=2);
colonLabel.grid(row=2, column=3)
inputMinutes.grid(row=2, column=3);
timeLabel.grid(row=2, column=1);

inputDate.grid(row=3, column=2);
dateLabel.grid(row=3, column=1);

firstLabel.grid(row=4, column=2);
secondLabel.grid(row=4, column=3);
thirdLabel.grid(row=4, column=4); 

photoLabel.grid(row=1, column=5, rowspan=3, columnspan=6, padx=10, pady=20);
photo2Label.grid(row=6, column=1, rowspan=3, columnspan=9, padx=(200, 200), pady=80);

findDestinations.grid(row=3, column=3, pady=20);
destinationLabel.grid(row=4, column=1, pady=20); 

# event-loop for userinput
fenster.mainloop()

