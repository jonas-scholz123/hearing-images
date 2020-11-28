# hearing-images

This application allows for images/videos to be converted into sound by weaving a 2-D pseudo Hilbert curve through the 2-D image to create a 1-D representation. Every pixel's position is then mapped to a sound frequency, and every pixel's brightness is mapped to an amplitude.

By inversely Fourier transforming, this spectrum of frequencies/amplitudes is turned into a time-basis audio signal which can be played through headphones.

This could potentially allow visually impaired people to substitute vision through audio signals.

This is based on the video: https://www.youtube.com/watch?v=3s7h2MHQtxc&t=461s&ab_channel=3Blue1Brown

Special requirements:

```
pip install hilbertcurve
```

To run:

```
python3 gui.py
```
