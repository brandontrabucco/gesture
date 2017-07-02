# Gesture
Gesture is an Android App that uses Long Short-term Memory (LSTM) to perceive motion much like the human eye. This app is trained to differentiate between six different human gestures: waving, beckoning, halt, no, thumbs up, and clapping. The app displays a set of probabilities for each gesture based on what the front camera sees.

# Long Short-term Memory
Long Short-term Memory is a kind of Recurrent neural Network that models human memory. LSTM is very useful because it can be trained to learn, remember, and forget dynamically. A detailed overview of LSTM can be found on <a href="https://en.wikipedia.org/wiki/Long_short-term_memory">wikipedia</a>. This app uses a varient of LSTM that uses a convolutional algorithm. Convolutional LSTM has been shown to achieve higher accuracy rates than standard LSTM.

# Convolutional Neural Networks
A Convolutional Neural Network (CNN) is characterized by shared weights and locally receptive fields. A CNN has a number of layers, and each layer is composed of filters--shared weight matrices. In the case of this app, the filter is a square matrix that is slid across a black and white camera image. Each weight is multiplied by a corrosponding pixel, and a sum is computed. Thus, the filter is convolved with the camera image. An overview of the CNN algorithm can be found on <a href="https://en.wikipedia.org/wiki/Convolutional_neural_network">wikipedia</a>.

CNNs are useful algorithms because they greatly reduce the number of weights to train, allow neural networks to train faster, and often allow them to learn more complex patterns. In the context of this app, a Convolutional LSTM is used, which speeds up video frame processing, and allows the app to learn as the app is being used.

# The Six Gestures
This Android App is designed to recognize six different gestures. These gestures were chosen for the spatio-temporal differences. Each gesture involves a different shape of the hand combined with a different motion. Each gesture of Outlined below.

Gesture 1: Waving (Motioning hand left and right, Palm facing outwards, All fingers extended)

Gesture 2: Beckoning (Gesturing towards oneself, Palm facing inwards, All fingers extended)

Gesture 3: Halt (Thrusting one's hand forward, Palm facing outwards, All fingers extended)

Gesture 4: No (Rotating one's hand left and right, Palm facing outwards, fist balled, one finder extended)

Gesture 5: Thumbs Up (Thrusting one's hand forward, Fist balled, Thumb raised upwards)

Gesture 6: Clapping (Bringing one's hands together, Palms facing towards opposing hand, Fingers extended)
