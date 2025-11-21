# TNIE - Tiny Neural Inference Engine
A minimal neural inference engine, written in pure C, designed for IoT devices and low-power systems.

📌 Why did I develop this demo?  

In the AI ​​world, most applications rely on heavyweight frameworks like TensorFlow, PyTorch, or ONNX Runtime. However, these tools are too complex and often unusable in IoT contexts, embedded systems, low-power devices, or hardware with very little memory (microcontrollers, wearables, industrial sensors).  
So my question was: is it possible to run a neural network without external libraries, using only pure C, with lightweight code that is easily portable to any device?  
This project was born to answer this need: to create a mini neural inference engine written from scratch in C, modular, lightweight and understandable, useful as a basis for real-world applications.


🎯 Objective of the project?   

Build a Neural Network Inference Engine in pure C that is:  
* modular and lightweight;  
* able to run inference on real-world inputs (e.g., sensor data);  
* designed to load model weights from external files (planned for next iteration);  
* supports Dense, ReLU, Sigmoid, Softmax layers;  
* requires no external libraries or dependencies;  
* portable, optimizable, easy to understand and deploy on embedded systems  


🧩 What does this engine do?  

Using just a few kilobytes of memory:  
1. Receives numerical input (e.g., values ​​from sensors, accelerometers, audio, temperature, etc.);  
2. Processes it using a feed-forward neural network;  
3. Returns a prediction or classification;  


🚀 Areas of use  
* IoT: gesture, motion, and presence recognition, ecc.;  
* Wearable: fall detection, fitness tracking, posture, ecc.;  
* Medical: vital signs analysis, patient monitoring, ecc.;  
* Industrial: predictive sensors, preventive maintenance, ecc.;  
* Robotics: obstacle detection, motor control, ecc.;  
* Home automation;  


🛠️ How can it be implemented?  
* on ARM / RISC-V microcontrollers;  
* on STM32, ESP32, Arduino boards;  
* integrated with Python for training predictive models;  


🔄 Current Status:  
* Neural inference engine fully working;  
* XOR model implemented via hardcoded weights;  
* Modular architecture ready for expansion;  

  
🔭 Next Steps (Roadmap):  
* Load model weights from JSON / binary file;  
* Python utility script to export trained models;  
* Quantization (float32 → int8) for low-power devices;  
* Benchmark and profiling tools;  
* Deployment on STM32 / ESP32 boards;  

