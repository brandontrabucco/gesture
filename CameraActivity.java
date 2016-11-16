package com.brandontrabucco.apps.gesture;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.ActionBar;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.util.Size;
import android.util.SparseIntArray;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.nio.ByteBuffer;
import java.util.Arrays;

/**
 * An example full-screen activity that shows and hides the system UI (i.e.
 * status bar and navigation/system bar) with user interaction.
 */
public class CameraActivity extends AppCompatActivity {
    /**
     * Whether or not the system UI should be auto-hidden after
     * {@link #AUTO_HIDE_DELAY_MILLIS} milliseconds.
     */
    private static final boolean AUTO_HIDE = true;

    /**
     * If {@link #AUTO_HIDE} is set, the number of milliseconds to wait after
     * user interaction before hiding the system UI.
     */
    private static final int AUTO_HIDE_DELAY_MILLIS = 3000;
    private static final int AUTO_HIDE_TIMEOUT_MILLIS = 60000;

    /**
     * Some older devices needs a small delay between UI widget updates
     * and a change of the status and navigation bar.
     */
    private static final int UI_ANIMATION_DELAY = 300;
    private final Handler mHideHandler = new Handler();
    private View mContentView;
    private TextView mClassificationView;
    private final Runnable mHidePart2Runnable = new Runnable() {
        @SuppressLint("InlinedApi")
        @Override
        public void run() {
            // Delayed removal of status and navigation bar

            // Note that some of these constants are new as of API 16 (Jelly Bean)
            // and API 19 (KitKat). It is safe to use them, as they are inlined
            // at compile-time and do nothing on earlier devices.
            mContentView.setSystemUiVisibility(View.SYSTEM_UI_FLAG_LOW_PROFILE
                    | View.SYSTEM_UI_FLAG_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                    | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                    | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                    | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);
        }
    };
    private FloatingActionButton mCameraAction;
    private FloatingActionButton mGestureOne;
    private FloatingActionButton mGestureTwo;
    private FloatingActionButton mGestureThree;
    private FloatingActionButton mGestureFour;
    private FloatingActionButton mGestureFive;
    private FloatingActionButton mGestureSix;

    private ActionBar mActionBar;
    private final Runnable mShowPart2Runnable = new Runnable() {
        @Override
        public void run() {
            // Delayed display of UI elements
            mActionBar.show();
            mCameraAction.show();
        }
    };
    private boolean mUIVisible;
    private boolean mMenuVisible;
    private final Runnable mHideRunnable = new Runnable() {
        @Override
        public void run() {
            hideUI();
        }
    };
    /**
     * Touch listener to use for in-layout UI controls to delay hiding the
     * system UI. This is to prevent the jarring behavior of controls going away
     * while interacting with activity UI.
     */
    private final View.OnTouchListener mDelayHideTouchListener = new View.OnTouchListener() {
        @Override
        public boolean onTouch(View view, MotionEvent motionEvent) {
            if (AUTO_HIDE) {
                delayedHide(AUTO_HIDE_DELAY_MILLIS);
            }
            return false;
        }
    };

    /**
     * Camera class resources and variables are initialized here
     */
    private static final String TAG = "CameraActivity";
    private TextureView textureView;
    private String cameraId;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession cameraCaptureSessions;
    protected CaptureRequest.Builder captureRequestBuilder;
    private Size imageDimension;
    private ImageReader imageReader;
    private static final int REQUEST_CAMERA_PERMISSION = 200;
    private Handler mBackgroundHandler;
    private HandlerThread mBackgroundThread;

    /**
     * This section is for the camera clas and its
     * respective functions and resources
     */

    /**
     * The Machine Learning Algorithm to process each video frame
     */
    private LSTMNeuralNetwork network;
    private double learningRate = 1.0;
    private double decayRate = 1.0;

    private boolean learningEnabled = false;
    private int learningTarget;
    private double[][] encodedTarget = {
            {.9, .1, .1, .1, .1, .1},
            {.1, .9, .1, .1, .1, .1},
            {.1, .1, .9, .1, .1, .1},
            {.1, .1, .1, .9, .1, .1},
            {.1, .1, .1, .1, .9, .1},
            {.1, .1, .1, .1, .1, .9}
    };

    TextureView.SurfaceTextureListener textureListener = new TextureView.SurfaceTextureListener() {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
            //open your camera here
            openCamera();
        }
        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) {
            // Transform you image captured size according to the surface width and height
        }
        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
            return false;
        }
        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface) {
        }
    };

    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            //This is called when the camera is open
            Log.e(TAG, "onOpened");
            cameraDevice = camera;
            startBackgroundThread();
            createCameraPreview();
        }
        @Override
        public void onDisconnected(CameraDevice camera) {
            cameraDevice.close();
            stopBackgroundThread();
        }
        @Override
        public void onError(CameraDevice camera, int error) {
            cameraDevice.close();
            cameraDevice = null;
            stopBackgroundThread();
        }
    };

    protected void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("Camera Background");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    protected void stopBackgroundThread() {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    protected void createCameraPreview() {
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            Surface surface = new Surface(texture);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            captureRequestBuilder.addTarget(surface);

            // Set the frame listener
            imageReader = ImageReader.newInstance(imageDimension.getWidth(), imageDimension.getHeight(), ImageFormat.YUV_420_888, 1);
            ImageReader.OnImageAvailableListener readerListener = new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    Image image = reader.acquireLatestImage();
                    ByteBuffer buffer = image.getPlanes()[0].getBuffer();
                    byte[] bytes = new byte[buffer.capacity()];
                    buffer.get(bytes);
                    process(bytes);
                    if (image != null) {
                        image.close();
                    }
                }
                private void process(byte[] bytes) {
                    // Do some image processing here

                    double[] data = new double[bytes.length];
                    for (int i = 0; i < bytes.length; i++) {
                        data[i] = ((double)((int)bytes[i] + 128)/256.0);
                    }
                    final double[] output = network.forward(data);

                    int classification = -1;
                    for (int i = 0; i < output.length; i++) {
                        if (classification == -1 && output[i] == 1) {
                            classification = i;
                        } else if (classification != -1 && output[i] == 1) {
                            classification = -1;
                            break;
                        }
                    }

                    final int result = classification + 1;

                    CameraActivity.this.runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            mClassificationView.setText(String.format("%.2f\n%.2f\n%.2f\n%.2f\n%.2f\n%.2f", output[0], output[1], output[2], output[3], output[4], output[5]));
                        }
                    });

                    if (learningEnabled) {
                       network.backward(encodedTarget[learningTarget]);
                    }
                }
            };
            imageReader.setOnImageAvailableListener(readerListener, mBackgroundHandler);
            if (imageReader.getSurface() == null) System.out.println("Test error, surface is null!");
            captureRequestBuilder.addTarget(imageReader.getSurface());

            cameraDevice.createCaptureSession(Arrays.asList(surface, imageReader.getSurface()), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                    //The camera is already closed
                    if (null == cameraDevice) {
                        return;
                    }
                    // When the session is ready, we start displaying the preview.
                    cameraCaptureSessions = cameraCaptureSession;
                    updatePreview();
                }

                @Override
                public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {
                    Toast.makeText(CameraActivity.this, "Configuration change", Toast.LENGTH_SHORT).show();
                }
            }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        Log.e(TAG, "is camera open");
        try {
            // Get the Camera properties
            cameraId = manager.getCameraIdList()[0];
            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            int numSizes = map.getOutputSizes(SurfaceTexture.class).length;
            System.out.println("Image Preview Tamplates : " + numSizes);
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[numSizes - 1];
            System.out.println("Image Dimensions: " + imageDimension.getWidth() + "x" + imageDimension.getHeight());

            // Configure the Neural Network
            int[] size = new int [] {(imageDimension.getWidth() * imageDimension.getHeight()), 6};
            network = new LSTMNeuralNetwork(size, learningRate, decayRate);

            // Add permission for camera and let user grant the permission
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(CameraActivity.this, new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, REQUEST_CAMERA_PERMISSION);
                return;
            }
            manager.openCamera(cameraId, stateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
        Log.e(TAG, "openCamera X");
    }

    protected void updatePreview() {
        if(null == cameraDevice) {
            Log.e(TAG, "updatePreview error, return");
        }
        captureRequestBuilder.set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO);
        try {
            cameraCaptureSessions.setRepeatingRequest(captureRequestBuilder.build(), null, mBackgroundHandler);

        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void closeCamera() {
        if (null != cameraDevice) {
            cameraDevice.close();
            cameraDevice = null;
        }
        if (null != imageReader) {
            imageReader.close();
            imageReader = null;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_camera);

        mUIVisible = true;
        mActionBar = getSupportActionBar();
        mContentView = findViewById(R.id.fullscreen_content);
        mClassificationView = (TextView)findViewById(R.id.classification);

        // Set up the user interaction to manually showUI or hideUI the system UI.
        mContentView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                toggleUI();
            }
        });

        // Upon interacting with UI controls, delay any scheduled hideUI()
        // operations to prevent the jarring behavior of controls going away
        // while interacting with the UI.

        textureView = (TextureView) findViewById(R.id.texture);
        textureView.setSurfaceTextureListener(textureListener);

        mCameraAction = (FloatingActionButton)findViewById(R.id.camera_action);
        mCameraAction.setOnTouchListener(mDelayHideTouchListener);
        mCameraAction.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                // do an action
                toggleMenu();
            }
        });

        mGestureOne = (FloatingActionButton)findViewById(R.id.gesture_one);
        mGestureTwo = (FloatingActionButton)findViewById(R.id.gesture_two);
        mGestureThree = (FloatingActionButton)findViewById(R.id.gesture_three);
        mGestureFour = (FloatingActionButton)findViewById(R.id.gesture_four);
        mGestureFive = (FloatingActionButton)findViewById(R.id.gesture_five);
        mGestureSix = (FloatingActionButton)findViewById(R.id.gesture_six);

        mGestureOne.setOnTouchListener(mDelayHideTouchListener);
        mGestureTwo.setOnTouchListener(mDelayHideTouchListener);
        mGestureThree.setOnTouchListener(mDelayHideTouchListener);
        mGestureFour.setOnTouchListener(mDelayHideTouchListener);
        mGestureFive.setOnTouchListener(mDelayHideTouchListener);
        mGestureSix.setOnTouchListener(mDelayHideTouchListener);

        mGestureOne.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_TIMEOUT_MILLIS);
                }

                learningEnabled = true;
                learningTarget = 0;
                return false;
            }
        });
        mGestureTwo.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_TIMEOUT_MILLIS);
                }

                learningEnabled = true;
                learningTarget = 1;
                return false;
            }
        });
        mGestureThree.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_TIMEOUT_MILLIS);
                }

                learningEnabled = true;
                learningTarget = 2;
                return false;
            }
        });
        mGestureFour.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_TIMEOUT_MILLIS);
                }

                learningEnabled = true;
                learningTarget = 3;
                return false;
            }
        });
        mGestureFive.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_TIMEOUT_MILLIS);
                }

                learningEnabled = true;
                learningTarget = 4;
                return false;
            }
        });
        mGestureSix.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View v, MotionEvent event) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_TIMEOUT_MILLIS);
                }

                learningEnabled = true;
                learningTarget = 5;
                return false;
            }
        });

        mGestureOne.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_DELAY_MILLIS);
                }

                learningEnabled = false;
            }
        });
        mGestureTwo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_DELAY_MILLIS);
                }

                learningEnabled = false;
            }
        });
        mGestureThree.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_DELAY_MILLIS);
                }

                learningEnabled = false;
            }
        });
        mGestureFour.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_DELAY_MILLIS);
                }

                learningEnabled = false;
            }
        });
        mGestureFive.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_DELAY_MILLIS);
                }

                learningEnabled = false;
            }
        });
        mGestureSix.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (AUTO_HIDE) {
                    delayedHide(AUTO_HIDE_DELAY_MILLIS);
                }

                learningEnabled = false;
            }
        });

        hideMenu();
        mMenuVisible = false;
    }

    @Override
    protected void onPostCreate(Bundle savedInstanceState) {
        super.onPostCreate(savedInstanceState);

        // Trigger the initial hideUI() shortly after the activity has been
        // created, to briefly hint to the user that UI controls
        // are available.
        delayedHide(100);
    }

    private void toggleUI() {
        if (mUIVisible) {
            hideUI();
        } else {
            showUI();
        }
    }

    private void toggleMenu() {
        if (mMenuVisible) {
            hideMenu();
        } else {
            showMenu();
        }
    }

    private void hideMenu() {
        mGestureOne.hide();
        mGestureTwo.hide();
        mGestureThree.hide();
        mGestureFour.hide();
        mGestureFive.hide();
        mGestureSix.hide();
        mMenuVisible = false;
    }

    private void showMenu() {
        mGestureOne.show();
        mGestureTwo.show();
        mGestureThree.show();
        mGestureFour.show();
        mGestureFive.show();
        mGestureSix.show();
        mMenuVisible = true;
    }

    private void hideUI() {
        // Hide UI first
        hideMenu();
        mActionBar.hide();
        mCameraAction.hide();
        mUIVisible = false;

        // Schedule a runnable to remove the status and navigation bar after a delay
        mHideHandler.removeCallbacks(mShowPart2Runnable);
        mHideHandler.postDelayed(mHidePart2Runnable, UI_ANIMATION_DELAY);
    }

    @SuppressLint("InlinedApi")
    private void showUI() {
        // Show the system bar
        mContentView.setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION);
        mUIVisible = true;

        // Schedule a runnable to display UI elements after a delay
        mHideHandler.removeCallbacks(mHidePart2Runnable);
        mHideHandler.postDelayed(mShowPart2Runnable, UI_ANIMATION_DELAY);

        delayedHide(AUTO_HIDE_DELAY_MILLIS);
    }

    /**
     * Schedules a call to hideUI() in [delay] milliseconds, canceling any
     * previously scheduled calls.
     */
    private void delayedHide(int delayMillis) {
        mHideHandler.removeCallbacks(mHideRunnable);
        mHideHandler.postDelayed(mHideRunnable, delayMillis);
    }
}
