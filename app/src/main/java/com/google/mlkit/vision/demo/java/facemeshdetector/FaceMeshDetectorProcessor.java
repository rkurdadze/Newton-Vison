/*
 * Copyright 2022 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.java.facemeshdetector;

import android.content.Context;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Task;
import com.google.android.odml.image.MlImage;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.java.VisionProcessorBase;
import com.google.mlkit.vision.demo.preference.PreferenceUtils;
import com.google.mlkit.vision.facemesh.FaceMesh;
import com.google.mlkit.vision.facemesh.FaceMeshDetection;
import com.google.mlkit.vision.facemesh.FaceMeshDetector;
import com.google.mlkit.vision.facemesh.FaceMeshDetectorOptions;

import java.util.ArrayList;
import java.util.List;

/**
 * Selfie Face Detector Demo.
 */
public class FaceMeshDetectorProcessor extends VisionProcessorBase<List<FaceMesh>> {
    private static final String TAG = "SelfieFaceProcessor";

    private final FaceMeshDetector detector;

    public FaceMeshDetectorProcessor(Context context) {
        super(context);
        FaceMeshDetectorOptions.Builder optionsBuilder = new FaceMeshDetectorOptions.Builder();
        if (PreferenceUtils.getFaceMeshUseCase(context) == FaceMeshDetectorOptions.BOUNDING_BOX_ONLY) {
            optionsBuilder.setUseCase(FaceMeshDetectorOptions.BOUNDING_BOX_ONLY);
        }

        detector = FaceMeshDetection.getClient(optionsBuilder.build());
    }

    @Override
    public void stop() {
        super.stop();
        detector.close();
    }

    @Override
    protected Task<List<FaceMesh>> detectInImage(InputImage image) {
        return detector.process(image);
    }

    @Override
    protected void onSuccess(@NonNull List<FaceMesh> faces, @NonNull GraphicOverlay graphicOverlay) {
        for (FaceMesh face : faces) {
            graphicOverlay.add(new FaceMeshGraphic(graphicOverlay, face));
        }

        if (faces.size()>0) {
            CustomEvent event = new CustomEvent(faces.get(0));
            for (CustomEventHandler listener : listeners) {
                listener.onCustomEvent(event);
            }
//            Log.e("#######", "detected!");
        }
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Face detection failed " + e);
    }






    private List<CustomEventHandler> listeners = new ArrayList<>();

    public void addCustomEventListener(CustomEventHandler listener) {
        listeners.add(listener);
    }

    public void removeCustomEventListener(CustomEventHandler listener) {
        listeners.remove(listener);
    }
}


