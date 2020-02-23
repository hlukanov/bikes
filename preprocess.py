# -*- coding: utf-8 -*-
"""
@author: jjoachims
"""

# import necessary packages
import csv, re, numpy as np, pickle;
from sklearn.preprocessing import OneHotEncoder;
from datetime import datetime;
import xlsxwriter
import xlwt

# set necessary paths and filenames
one_hot_path = './models/one_hot_encoder.pkl';
csv_path = './data/train_test_data.csv';
raw_data_path = './data/route_data.csv';

# specify excelsheets to import from
workbook1 = xlsxwriter.Workbook('./data/dict.xlsx');
sheet = workbook1.add_worksheet("conversion");
workbook3 = xlwt.Workbook(encoding="utf-8");
sheet3 = workbook3.add_sheet("stations");

# load the data file
reader = csv.reader(open(raw_data_path, 'r', encoding='utf-8'), delimiter=';');
next(reader); # skip the header row

# one-hot encoder
enc = OneHotEncoder(handle_unknown='ignore');

# create empty arrays and lists for later filling
classes_count = {};
features = [];
classes = [];
excluded = [];
names = {};
stations = [];
stationsToId = {};
idToStations = {};

# go through every row of the data to extract the necessary features
for row in reader:
    start_station = re.sub('\s*/+\s*', '/', row[10]);# regex expression 
    end_station = re.sub('\s*/+\s*', '/', row[12]);
    date_from = row[5];
    city = row[15];

    # exclude empty cells and rows referencing other cities
    if start_station == '' or end_station == '' or city != 'Hamburg':
        continue;

    start_station_id = str(int(float(row[11])));#transfer float to integer to string
    end_station_id = str(int(float(row[13])));

    if end_station_id in excluded:
        continue;

    try:
        startDate = datetime.strptime(date_from, '%Y-%m-%d %H:%M:%S');# convert string to datetime
    except TypeError:
        print("Incorrect date format");
        continue;
    
    # fill dicts for later id to station and station to id conversion
    if start_station not in stationsToId:
        stationsToId.update({start_station: start_station_id});
        
    if end_station not in stationsToId:
        stationsToId.update({end_station: end_station_id});
        
    if start_station not in idToStations:
        idToStations.update({start_station_id: start_station});
        
    if end_station not in idToStations:
        idToStations.update({end_station_id: end_station});
    
    # the y or labels for the training
    if end_station_id not in classes:
        classes.append(end_station_id);

    # count number of appearances in the code, how many times the answer is end station
    if end_station_id not in classes_count:
        classes_count[end_station_id] = 0;
    
    if start_station not in stations:
        stations.append(start_station);

    if end_station not in stations:
        stations.append(end_station);

    # add one to the count of samples per class
    classes_count[end_station_id] += 1;

    # get at most 10 000 samples per class, so the training data is balanced
    # the border was set between 1000 and 10000 so as not to introduce bias
    if classes_count[end_station_id] > 10000:
        excluded.append(end_station_id);

    # get unique station names
    if start_station_id not in names:
        names[start_station_id] = start_station;
    if end_station_id not in names:
        names[end_station_id] = end_station;
        
    # classifier doesn't know which features are separate and which are important, 
    # if you have numbers that are bigger than others then you're introducing a bias
    # scaling between 0 and 1 allows you to compare values because they have the same 
    # dimensions, so all features are equally important
    dayOfWeek = round( startDate.weekday()/6., 2 ); # day of week normalized between 0 and 1
    hour = round( startDate.hour/23., 2 ); # hours normalized between 0 and 1
    minute = startDate.minute;

    # minutes rounded and normalized between 0 and 1
    if minute < 15:
        minute = 0.;
    elif minute < 30:
        minute = 0.25;
    elif minute < 45:
        minute = 0.50;
    elif minute < 60:
        minute = 0.75;
        
    # training data with extracted features but normalized
    features.append([dayOfWeek, hour, minute, start_station_id, end_station_id]);

# go to last values of row in array, if value in dict and if value has more than 1000 features
for i in range(len(features)-1,0,-1):
    if features[i][-1] in classes_count and classes_count[features[i][-1]] < 1000:
        del features[i];

print('Data preprocessed');

# encoder only takes specific shape so array needs to be split into single arrays with single ids
classes = np.array(classes).reshape(-1,1).tolist();

print('Training data includes: {0} classes'.format(len(classes)));

# teach one hot encoder how to transform ids to vectors to bring it into a shape that has a 
# lot of zeros and one 1, this gets rid of the order of classes
enc.fit(classes);

# free a bit of memory
del classes;

print('Features encoded');

# save the one hot encoder, so that the same vector persists and the order of classes isn't 
# scrambled, otherwise there is the same output for different classes
with open(one_hot_path, 'wb') as f:
    pickle.dump(enc, f);

# save the preprocessed data
with open(csv_path, mode='w') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL);

    for k in range(0,len(features)):
        csv_writer.writerow(features[k]);

# save the stations
for i,e in enumerate(stations):
    sheet3.write(i,0,e);

name = './data/stations.xls';
workbook3.save(name);

# save dict to sheet of a workbook
row = 0;
col = 0;

for key in stationsToId.keys():
    sheet.write(row, col, key);
    sheet.write(row, col+1, stationsToId[key]);
    row += 1;

print('Dicts saved');
     
workbook1.close();

print('Done!');
