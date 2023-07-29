# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Folders available and what's in them
 - artifacts
   - It stores the temporary data required for the code
 - datasets
   - It contains the raw housing data which is cleaned and used for codes
 - dist
   - It contains the files neccessary to install packages made from this code
 - docs
   - It contains the documentation of this project
 - logs
   - It contains the logs for the basic codes
 - notebook
   - It contains all the .pynb file
 - src
   - It contains the codes for the project
 - test
   - It contains the test codes to verify all files and folders are present 

## How to run the code
 - Open miniconda prompt
 - Change directory to project folder
 - Type the below line to replicate the **environment** created
   - conda env create --file env.yml
 - Type any one of the below line to **install package**
   - pip install dist/median_housing_value_prediction-0.3-py3-none-any.whl
   - pip install dist/median_housing_value_prediction-0.3.tar.gz
 - To **test** the installation, follow the below steps
   - Change directory to test/
   - Execute below codes to check if the packages and files are all available
     - python test_unit.py
     - python test_package.py
   - After execution, if it throws no error, then it is safe to follow next steps, else download the files again
 - Now change directory to src/
 - Type the below line to **execute the code**
   - python ingest_data.py --> to create the dataset for further process, else default dataset will be used
   - python train.py --> to train the datasets available, else default trained model will be used
   - python score.py --> to display the scores of different models used
 - Alternatively 
   - if you want to change log path for any code
     - python "code name".py --logpath "log path directory"
   - if you want to change artifacts path for any code
     - python "code name".py -o "artifacts path directory"
 - To open it with jupyter notebook, just o to project folder and type the following in the prompt
   - jupyter notebook
   Then copy the url from the prompt into a web browser to use jupyter notebook
 - To check flow of the code using **mlflow**, follow the below steps
   - Open a new miniconda prompt and change directory to the project folder
   - Activate the conda environment mle-dev
   - Now paste the below script to run the server
     - mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host localhost --port 5000
   - After running the command, open your browser and paste the url created on the miniconda prompt. Now you may see the UI of Mlflow.
   - Open a new miniconda prompt and change directory to the src folder under project folder
   - Activate the conda environment mle-dev
   - Run the main.py file by running the below command
     - python main.py
   - Wait for a few minutes, and after the code runs successfully, you can check at the directory described in the prompt for the parameters and metrics (or) you can refresh the website and check the recent run to check for result
 - To use docker (Note: You need to be the root user),
   - Start up the server in one prompt by typing
     - sudo dockerd
   - To create image of the project using dockerfile, go to the project folder and type
     - docker build -t "name of image" .
   - Alternatively, you can pull image from hub.docker.com by using the below lines
     - docker pull tadavish/housing_price_prediction
   - To use this image, type
     - docker run -d -p 5000:5000 --name "name of container" "name of image"
   - Now, a server will be stated at http://localhost:5000, check it for the Mlflow UI
   - Change directory to src and type the below line to run the code
     - docker exec -it "name of container" python main.py
