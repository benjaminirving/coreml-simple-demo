//
//  main.swift
//  CmdSand
//
//  Created by Benjamin Irving on 22/08/2017.
//  Copyright © 2017 Benjamin Irving. All rights reserved.
//

import Vision
import CoreML

//import Cocoa
//import Foundation
//import CoreImage

// Handler for model
func output_handler(request: VNRequest, error: Error?) {
    guard let results = request.results as? [VNClassificationObservation]
        else { fatalError("Unable to process") }
    print("That looks like a:")
    for class1 in results {
        if (class1.confidence > 0.05) {
            // Only show greater than 5%
            print(class1.identifier,
                round(class1.confidence*100))
        }
    }
}

// arg[1] should be in path
let a = CommandLine.arguments
let size = a.count// Otherwise throw an error
if (size != 2) {
    print("Incorrect number of arguments")
    exit(0)
}

// Load image from file
//let inputURL = URL(fileURLWithPath: "/Users/benjamin/Code/Sandbox/CmdSand/cmdLabel/elephant.jpg")
let inputURL = URL(fileURLWithPath: a[1])
let inputImage = CIImage(contentsOf: inputURL)

// Easier way to do it but not supported in keras
guard let model = try? VNCoreMLModel(for: ResNet50().model) else {
    fatalError("can't load ML model")
}

// Process image
let request = VNCoreMLRequest(model: model, completionHandler: output_handler)
let handler = VNImageRequestHandler(ciImage: inputImage!)
try handler.perform([request])



