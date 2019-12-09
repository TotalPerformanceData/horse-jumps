# horse-jumps
Models trained at TPD to classify sequences of lightweight tracker observations as either a Jump or Not a Jump. Typically Speed and Stride Length near obstacles form the following time series,

<img src="/images/speed.png" alt="Speed near Obstacles" width="400" height="150">
<img src="/images/strideLength.png" alt="Stride Length near Obstacles" width="400" height="150">

Both a Live model using previous 8.5 seconds of data points as well as a Retrospective model additionally using the next 4s after some examined timestamp. Live model applications include live trading bet management, and powering race animations.

Models are trained using keras with tensorflow backend.

The validation score of the Live model was 98.07%. There were 17 (1.44%) instances of a false positive in the validation data, where a positive result is defined as a prediction greater than 0.75. Predictions are made by horse so by combining nearby runners the probability of 2 or more runners supplying a false positive in some regoin is remote.

Fences of known locations as professionally surveyed at a handful of racecourses and GPS data points from the TPD Points Feed at 2Hz were used to generate the data set. Testing on Live recordings of the data (at 1Hz and interpolated to 2Hz to match model input requirement)  showed comparable accuracy.

Predictions on my machine takes around 7ms + N\*0.07ms, for N predictions passed to function. Entry to the predict() function seems to be the main restriction on speed and I expect would be faster if converted to another language and/or compiled though many of the methods appear to be platform specific and require a bunch of tinkering, but I welcome feedback on that front.

Input data for the model is supplied, as improvements on the model would be welcome.

Heatmaps can be produced by averaging the scores of nearby Lat-Lon points. The output of which for a couple of races over fences are shown below. The heatmaps show as more red for average scores which are more confident of close proximity to an obstacle, and more blue otherwise.

<img src="/images/WorcesterRetro.png" alt="Retrospective model applied to a race a Worcester" width="400" height="300"> <img src="/images/WorcesterLive.png" alt="Live model applied to a race a Worcester" width="400" height="300">

<img src="/images/UttoxeterRetro.png" alt="Retrospective model applied to a race a Uttoxeter" width="400" height="300"> <img src="/images/UttoxeterLive.png" alt="Live model applied to a race a Uttoxeter" width="400" height="300">

The performance of the model on Fences is very good.

Fences are a little bigger than Hurdles so the characteristics of the observation attributes when approaching an obstacle are generally clearer and easier to predict. Also, the training dataset only contains observations from Fence races because currently surveyed data for the Hurdles is very difficult to source accurately since they move around the course meeting to meeting so it's likely that there will be occaisions whereby the horses are sufficiently fluent over the obstacle such that the model calls it Flat. That being said, the perforamnce of the model on Hurdle races is still useable for some applications as indicated by the heatmap below.

<img src="/images/SouthwellRetroHurdle.png" alt="Retrospective model applied to a Hurdle race a Southwell" width="400" height="300"> <img src="/images/SouthwellLiveHurdle.png" alt="Live model applied to a Hurdle race a Southwell" width="400" height="300">

I've side noted that often the better class, 3+, horses tend to produce more confusion over Hurdles than that of lower class horses which could be a consideration in applications.

