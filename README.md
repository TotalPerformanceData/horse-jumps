# horse-jumps
Models trained at TPD to classify sequences of lightweight tracker observations as either a Jump or Not a Jump

Both a Live model using previous 8.5 seconds of data points as well as a Retrospective model additionally using the next 4s after some examined timestamp. Live model applications include live trading bet management, and powering race animations.

The validation score of the Live model was 98.07%. There were 17 (1.44%) instances of a false positive in the validation data, where a positive result is defined as a prediction greater than 0.75. Predictions are made by horse so by combining nearby runners the probability of 2 or more runners supplying a false positive in some regoin is remote.

Fences of known locations as professionally surveyed at a handful of racecourses and GPS data points from the TPD Points Feed at 2Hz were used to generate the data set. Testing on Live recordings of the data (at 1Hz and interpolated to 2Hz to match model input requirement)  showed comparable accuracy.

Input data for the model is supplied, improvements on the model would be welcome.
