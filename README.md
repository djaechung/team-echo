# team-echo
Here we apply deep learning models to predict ejection fraction (EF) from echocardiograms, which are ultrasound readings of the heart. This is both a binary classification problem, to classify echocardiograms as being healthy (EF >= 40%) and unhealthy (EF < 40%), and a regression problem, to predict the actual EF value.

Our data comes from the EchoNet dataset, which can be found at this link:
https://echonet.github.io/dynamic/

Our student team implemented two approaches:
* Our SwinTransformer achieves a test AUC score of 0.88 and a test accuracy of 90% on the binary classification task
* Our R3D model achieving a test AUC of 0.92 and a test accuracy of 92%, also on the binary classification task

Another version of this repository can be found on contributor `VasuKaker`'s Github:
https://github.com/VasuKaker/ML4H_Team_Echo

Contents:
* `data_modifiers` - scripts for distributing and balancing echocardiogram data
* `datasets` - contains echocardiogram video data. Open distribution of the EchoNet dataset is not permitted, so for this public repository we have redacted this data.
* `helpers` - helper functions
* `models` - contains both binary classification (EF above or below 40%) and continuous EF regressor models
* `FileList.csv` - table containing metadata on echocardiogram videos, including EF labels, framelength, and assigned train-val-test split

Code Contributors:
> Vasu Kaker `VasuKaker` <br /> Daniel Chung `djaechung` <br /> Mindy Somin Lee `mindyslee` <br /> Yongyi Zhao

Clinical Contributors:
> Irbaz Riaz <br /> Sudheesha Perera <br /> Prabhu Sasankan <br /> George Tang <br /> Kpodonu Jacques <br /> Brigitte Kazz <br /> Leo Anthony Celi

Technical Consultation:
> Po-Chih Kuo
