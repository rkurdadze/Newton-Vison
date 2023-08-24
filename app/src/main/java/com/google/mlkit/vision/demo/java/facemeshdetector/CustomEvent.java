package com.google.mlkit.vision.demo.java.facemeshdetector;

import com.google.mlkit.vision.facemesh.FaceMesh;

public class CustomEvent {
    private FaceMesh face;

    public CustomEvent(FaceMesh value) {
        this.face = value;
    }

    public FaceMesh getFace() {
        return face;
    }
}
