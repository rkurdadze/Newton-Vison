/*
 * Copyright 2020 Google LLC. All rights reserved.
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

package com.google.mlkit.vision.demo.java;

import static android.provider.Telephony.Mms.Part.FILENAME;

import android.content.Intent;
import android.os.Bundle;

import androidx.appcompat.app.AppCompatActivity;

import android.os.Message;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.ArrayAdapter;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import com.google.android.gms.common.annotation.KeepName;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.demo.CameraSource;
import com.google.mlkit.vision.demo.CameraSourcePreview;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.R;
import com.google.mlkit.vision.demo.java.facedetector.FaceDetectorProcessor;
import com.google.mlkit.vision.demo.java.facemeshdetector.CustomEvent;
import com.google.mlkit.vision.demo.java.facemeshdetector.CustomEventHandler;
import com.google.mlkit.vision.demo.java.facemeshdetector.FaceMeshDetectorProcessor;
import com.google.mlkit.vision.demo.preference.SettingsActivity;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceContour;
import com.google.mlkit.vision.facemesh.FaceMesh;
import com.google.mlkit.vision.facemesh.FaceMeshPoint;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Live preview demo for ML Kit APIs.
 */
@KeepName
public final class LivePreviewActivity extends AppCompatActivity
        implements OnItemSelectedListener, CompoundButton.OnCheckedChangeListener {
    private static final String FACE_DETECTION = "Face Detection";
    private static final String FACE_MESH_DETECTION = "Face Mesh Detection (Beta)";

    private static final String TAG = "LivePreviewActivity";

    private CameraSource cameraSource = null;
    private CameraSourcePreview preview;
    private GraphicOverlay graphicOverlay;
    private String selectedModel = FACE_MESH_DETECTION;

    private FaceMesh face;
    private FaceMeshDetectorProcessor fm;
    private TextView probabilityText;

    private FloatingActionButton saveMeshBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate");

        setContentView(R.layout.activity_vision_live_preview);

        probabilityText = findViewById(R.id.probabilityText);
        saveMeshBtn = findViewById(R.id.saveMeshBtn);
        saveMeshBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (face == null) return;
                JSONObject faceMeshData = new JSONObject();
                try {
                    List<FaceMeshPoint> contourPoints = face.getAllPoints();
                    // Create a JSON array to store the contour points
                    JSONArray contourPointsArray = new JSONArray();
                    // Loop through each contour point and add it to the JSON array
                    for (int k = 0; k < contourPoints.size(); k++) {
                        FaceMeshPoint point = contourPoints.get(k);
                        JSONObject pointObject = new JSONObject();
                        try {
                            pointObject.put("x", point.getPosition().getX());
                            pointObject.put("y", point.getPosition().getY());
                            pointObject.put("z", point.getPosition().getZ());
                            contourPointsArray.put(pointObject);
                        } catch (JSONException e) {
                            Log.e(TAG, "Error adding contour point to JSON array: " + e.getMessage());
                        }
                    }
                    // Add the contour points array to the face mesh data JSON object
                    try {
                        faceMeshData.put("", contourPointsArray);
                    } catch (JSONException e) {
                        Log.e(TAG, "Error adding contour points array to JSON object: " + e.getMessage());
                    }

                } catch (Exception e) {
                }

                // Write the face mesh data to a JSON file
                File file = new File(LivePreviewActivity.this.getFilesDir(), "my.txt");
                try {
                    FileWriter writer = new FileWriter(file);
                    writer.write(faceMeshData.toString());
                    writer.close();
                    Toast.makeText(LivePreviewActivity.this, "Saved OK!", Toast.LENGTH_SHORT).show();
                } catch (Exception e) {
                    Toast.makeText(LivePreviewActivity.this, "Error writing file", Toast.LENGTH_SHORT).show();
                }


            }
        });

        preview = findViewById(R.id.preview_view);
        if (preview == null) {
            Log.d(TAG, "Preview is null");
        }
        graphicOverlay = findViewById(R.id.graphic_overlay);
        if (graphicOverlay == null) {
            Log.d(TAG, "graphicOverlay is null");
        }

        Spinner spinner = findViewById(R.id.spinner);
        List<String> options = new ArrayList<>();
        options.add(FACE_MESH_DETECTION);
        options.add(FACE_DETECTION);


        // Creating adapter for spinner
        ArrayAdapter<String> dataAdapter = new ArrayAdapter<>(this, R.layout.spinner_style, options);
        // Drop down layout style - list view with radio button
        dataAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        // attaching data adapter to spinner
        spinner.setAdapter(dataAdapter);
        spinner.setOnItemSelectedListener(this);

        ToggleButton facingSwitch = findViewById(R.id.facing_switch);
        facingSwitch.setOnCheckedChangeListener(this);

        ImageView settingsButton = findViewById(R.id.settings_button);
        settingsButton.setOnClickListener(
                v -> {
                    Intent intent = new Intent(getApplicationContext(), SettingsActivity.class);
                    intent.putExtra(
                            SettingsActivity.EXTRA_LAUNCH_SOURCE, SettingsActivity.LaunchSource.LIVE_PREVIEW);
                    startActivity(intent);
                });

        createCameraSource(selectedModel);
    }

    public double compareFaceMeshData() {
        double similarityProbability = 0.0;
        try {
            // Read the saved face mesh data from the JSON file
            File file = new File(LivePreviewActivity.this.getFilesDir(), "my.txt");
            FileInputStream inputStream = new FileInputStream(file);
            byte[] buffer = new byte[(int) file.length()];
            inputStream.read(buffer);
            String savedData = new String(buffer);

            // Parse the saved face mesh data from the JSON string
            JSONObject savedFaceMeshData = new JSONObject(savedData);

            // Compare the saved face mesh data with the face mesh data from memory
            double similarityScore = 0.0;
            double totalPoints = 468.0;

            List<FaceMeshPoint> contourPoints = face.getAllPoints();
//                int a=0;
            for (int k = 0; k < contourPoints.size(); k++) {
                FaceMeshPoint point = contourPoints.get(k);
                JSONObject savedPointObject = savedFaceMeshData.getJSONArray(String.valueOf("")).getJSONObject(k);
                double savedX = savedPointObject.getDouble("x");
                double savedY = savedPointObject.getDouble("y");
                double savedZ = savedPointObject.getDouble("z");
                double x = point.getPosition().getX();
                double y = point.getPosition().getY();
                double z = point.getPosition().getZ();
                double distanceX = Math.abs(savedX - x) / Math.max(savedX, x);
                double distanceY = Math.abs(savedY - y) / Math.max(savedY, y);
                double distanceZ = Math.abs(savedZ - z) / Math.max(savedZ, z);
                double similarityX = Math.max(0.0, 1.0 - distanceX);
                double similarityY = Math.max(0.0, 1.0 - distanceY);
                double similarityZ = Math.max(0.0, 1.0 - distanceZ);
                similarityScore += (similarityX + similarityY + similarityZ) / 3.0;
            }




            similarityProbability = similarityScore / totalPoints * 100.0;
        } catch (Exception e) {
            Log.e(TAG, "Error comparing face mesh data: " + e.getMessage());
        }
        return similarityProbability;
    }

    @Override
    public synchronized void onItemSelected(AdapterView<?> parent, View view, int pos, long id) {
        // An item was selected. You can retrieve the selected item using
        // parent.getItemAtPosition(pos)
        selectedModel = parent.getItemAtPosition(pos).toString();
        Log.d(TAG, "Selected model: " + selectedModel);
        preview.stop();
        createCameraSource(selectedModel);
        startCameraSource();
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {
        // Do nothing.
    }

    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
        Log.d(TAG, "Set facing");
        if (cameraSource != null) {
            if (isChecked) {
                cameraSource.setFacing(CameraSource.CAMERA_FACING_FRONT);
            } else {
                cameraSource.setFacing(CameraSource.CAMERA_FACING_BACK);
            }
        }
        preview.stop();
        startCameraSource();
    }

    private void createCameraSource(String model) {
        // If there's no existing cameraSource, create one.
        if (cameraSource == null) {
            cameraSource = new CameraSource(this, graphicOverlay);
        }

        try {
            switch (model) {

                case FACE_DETECTION:
                    Log.i(TAG, "Using Face Detector Processor");
                    cameraSource.setMachineLearningFrameProcessor(new FaceDetectorProcessor(this));
                    break;


                case FACE_MESH_DETECTION:
                    fm = new FaceMeshDetectorProcessor(this);
                    cameraSource.setMachineLearningFrameProcessor(fm);
                    fm.addCustomEventListener(new CustomEventHandler() {
                        @Override
                        public void onCustomEvent(CustomEvent event) {
                            face = event.getFace();
                            double res = compareFaceMeshData();
                            probabilityText.setText(String.valueOf(res));
//                            if (toast == null || toast.getView().getWindowVisibility() != View.VISIBLE) {

//                                Toast.makeText(LivePreviewActivity.this, String.valueOf(res), Toast.LENGTH_SHORT).show();
//                                this.toast = Toast.makeText(LivePreviewActivity.this, String.valueOf(res), Toast.LENGTH_LONG);
//                                toast.show();
//                            }
                        }
                    });
                    break;
                default:
                    Log.e(TAG, "Unknown model: " + model);
            }
        } catch (RuntimeException e) {
            Log.e(TAG, "Can not create image processor: " + model, e);
            Toast.makeText(
                            getApplicationContext(),
                            "Can not create image processor: " + e.getMessage(),
                            Toast.LENGTH_LONG)
                    .show();
        }
    }

    /**
     * Starts or restarts the camera source, if it exists. If the camera source doesn't exist yet
     * (e.g., because onResume was called before the camera source was created), this will be called
     * again when the camera source is created.
     */
    private void startCameraSource() {
        if (cameraSource != null) {
            try {
                if (preview == null) {
                    Log.d(TAG, "resume: Preview is null");
                }
                if (graphicOverlay == null) {
                    Log.d(TAG, "resume: graphOverlay is null");
                }
                preview.start(cameraSource, graphicOverlay);
            } catch (IOException e) {
                Log.e(TAG, "Unable to start camera source.", e);
                cameraSource.release();
                cameraSource = null;
            }
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");
        createCameraSource(selectedModel);
        startCameraSource();
    }

    /**
     * Stops the camera.
     */
    @Override
    protected void onPause() {
        super.onPause();
        preview.stop();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (cameraSource != null) {
            cameraSource.release();
        }
    }
}
