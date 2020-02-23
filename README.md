# bikes
Python 3.5.2 or newer is necessary

For running the program:
1. Download archive from GitHub
2. Extract archive
3. Download booking data from "https://data.deutschebahn.com/dataset/data-call-a-bike/resource/0fcce4dd-7fc6-43f8-a59c-983a7945f8ba"
4. Extract archive "OPENDATA_BOOKING_CALL_A_BIKE.csv", rename to route_data.csv and save in extracted bikes archive under subfolder "data"
5. Open a terminal and navigate to the extracted archive with "cd PATH/TO/FOLDER"
6. Install required dependencies with "pip install -r requirements.txt"
7. Run preprocess.py in an adequate run-time environment

For using the interface:
1. Execute all steps for running the program
2. Run interface.py
3. Enlarge window
4. Feed values to your hearts content

For testing/training the model:
1. Execute all steps for running the program
2. In the terminal, type "python model.py" for testing accuracy or "python model.py to train the model (if you just want to test the model you can also just run the model.py script directly)
3. See console for prediction accuracy of Support Vector Machine or wait until a new training model is created

(Note: This program runs with a pretrained model, which is used for the interface.py for demonstration purposes and to get the accuracy on the test set. It is not necessary to train the model again, but feel free to do so. It may take several days to train on any regular laptop.)
