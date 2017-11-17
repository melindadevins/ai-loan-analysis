# Loan Application Analysis Using Machine Learning and Deep Learning

### Ellie Mae Hatckathon 2017
### Team Crystal Ball

Analyze loan applications, and predict approval probablity using machine learning and deep learning algorithms

To run the notebook:

1. Install Anaconda
2. On PC, create environment *em_hack* from environment_pc.yaml by running command:

        conda env create -f environment_pc.yaml
        
   On Mac, create environment *em_hack* from environment_mac.yaml by running command:

        conda env create -f environment_mac.yaml
          
3. Activate the environment by running command: 

        activate em_hack
        
4. Launch notebook by running command: 

        jupyter notebook
        
5. Unzip all zip files under *data* directory before run any jupyter notebook files

6. The following is the content of the project:

*/asset*    Images, and slides used in Ellie Mae Hackthon 2017

*/data*     The input data files, including pre-processed data files

*/models*   Trained models

*/server*   Flask server that hosts REST API for real time prediction using trained models

