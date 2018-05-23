package edu.imac.nutc.tensorflowtest;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;

/**
 * Created by cheng on 2018/1/26.
 */

public class YoloActivity extends AppCompatActivity {
    private static final String TAG = YoloActivity.class.getName();
    private static final String MODEL_FILE = "file:///android_asset/tiny-yolo-voc2.pb";
    private static final String INPUT_NODE = "input";
    private static final String OUTPUT_NODE = "output";
    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 5;

    private static final int NUM_CLASSES = 20;

    private static final int NUM_BOXES_PER_BLOCK = 5;

    // TODO(andrewharp): allow loading anchors and classes
    // from files.
    private static final double[] ANCHORS = {
            1.08, 1.19,
            3.42, 4.41,
            6.63, 11.38,
            9.42, 5.11,
            16.62, 10.52
    };
    private static final String[] LABELS = {
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor"
    };
    private static int INPUT_SIZE = 416;
    private TensorFlowInferenceInterface inferenceInterface;
    private int[] intValues;
    private float[] inputFloat;
    private int gridWidth;
    private int gridHeight;
    private int blockSize=32;
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_inception);

    }

    private void init() {
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
        intValues = new int[INPUT_SIZE * INPUT_SIZE];
        inputFloat = new float[INPUT_SIZE * INPUT_SIZE * 3];
//        convertBitmapToByteBuffer(BitmapFactory.decodeResource(getResources(), R.drawable.image));
        convertBitmapToByteBuffer(getBitmapFromAsset(this, "bird416.png"));
    }

    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        HashMap<Object,String> resultMap=new HashMap();
        resultMap.put("confidence","0");
        Log.d(TAG, "convertBitmapToByteBuffer: " + bitmap.getWidth());
        Log.d(TAG, "convertBitmapToByteBuffer: " + bitmap.getHeight());
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        for (int i = 0; i < intValues.length; ++i) {
            final float r=((intValues[i] >> 16) & 0xFF)/255.0f;
            final float g=((intValues[i] >> 8) & 0xFF)/255.0f;
            final float b=(intValues[i] & 0xFF)/255.0f;
            inputFloat[i * 3 + 0] = r;
            inputFloat[i * 3 + 1] = g;
            inputFloat[i * 3 + 2] = b;
        }
        gridWidth = bitmap.getWidth() / blockSize;
        gridHeight = bitmap.getHeight() / blockSize;
        inferenceInterface.feed(INPUT_NODE, inputFloat, 1, INPUT_SIZE, INPUT_SIZE, 3);
        final float[] output =
                new float[gridWidth * gridHeight * (NUM_CLASSES + 5) * NUM_BOXES_PER_BLOCK];
        float result=0;
        int num=0;
        inferenceInterface.run(new String[]{OUTPUT_NODE});
        inferenceInterface.fetch(OUTPUT_NODE, output);
        for (int y = 0; y < gridHeight; ++y) {
            for (int x = 0; x < gridWidth; ++x) {
                for (int b = 0; b < NUM_BOXES_PER_BLOCK; ++b) {
                    final int offset =
                            (gridWidth * (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5))) * y
                                    + (NUM_BOXES_PER_BLOCK * (NUM_CLASSES + 5)) * x
                                    + (NUM_CLASSES + 5) * b;

                    final float xPos = (x + expit(output[offset + 0])) * blockSize;
                    final float yPos = (y + expit(output[offset + 1])) * blockSize;

                    final float w = (float) (Math.exp(output[offset + 2]) * ANCHORS[2 * b + 0]) * blockSize;
                    final float h = (float) (Math.exp(output[offset + 3]) * ANCHORS[2 * b + 1]) * blockSize;
                    final RectF rect =
                            new RectF(
                                    Math.max(0, xPos - w / 2),
                                    Math.max(0, yPos - h / 2),
                                    Math.min(bitmap.getWidth() - 1, xPos + w / 2),
                                    Math.min(bitmap.getHeight() - 1, yPos + h / 2));
                    final float confidence = expit(output[offset + 4]);

                    int detectedClass = -1;
                    float maxClass = 0;

                    final float[] classes = new float[NUM_CLASSES];
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        classes[c] = output[offset + 5 + c];
                    }
                    softmax(classes);

                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        if (classes[c] > maxClass) {
                            detectedClass = c;
                            maxClass = classes[c];
                        }
                    }
                    final float confidenceInClass = maxClass * confidence;
                    if (confidenceInClass > 0.01) {
                        if(confidenceInClass>Float.valueOf(resultMap.get("confidence"))){
                            resultMap.put("xPos",String.valueOf(xPos));
                            resultMap.put("yPos",String.valueOf(yPos));
                            resultMap.put("width",String.valueOf(w));
                            resultMap.put("height",String.valueOf(h));
                            resultMap.put("label",String.valueOf(LABELS[detectedClass]));
                            resultMap.put("confidence",String.valueOf(confidenceInClass));
                        }
                        Log.e("xPos",xPos+"");
                        Log.e("yPos",yPos+"");
                        Log.e("width",w+"");
                        Log.e("height",h+"");
                        Log.e("label",LABELS[detectedClass]);
                        Log.e("confidence",confidenceInClass+"");
                    }
                }
            }
        }
//        Log.e("xPos", Math.max(0, Float.valueOf(resultMap.get("xPos")) - Float.valueOf(resultMap.get("width")) / 2)+"");
//        Log.e("yPos",Math.max(0, Float.valueOf(resultMap.get("yPos")) - Float.valueOf(resultMap.get("height")) / 2)+"");
//        Log.e("width",Math.min(bitmap.getWidth() - 1, Float.valueOf(resultMap.get("xPos")) + Float.valueOf(resultMap.get("width")) / 2)+"");
//        Log.e("height",Math.min(bitmap.getWidth() - 1, Float.valueOf(resultMap.get("yPos")) + Float.valueOf(resultMap.get("height")) / 2)+"");
//        Log.e("label",resultMap.get("label")+"");
//        Log.e("confidence",resultMap.get("confidence")+"");
    }

    @Override
    protected void onPostResume() {
        super.onPostResume();
        init();
    }
    private float expit(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }

    private void softmax(final float[] vals) {
        float max = Float.NEGATIVE_INFINITY;
        for (final float val : vals) {
            max = Math.max(max, val);
        }
        float sum = 0.0f;
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = (float) Math.exp(vals[i] - max);
            sum += vals[i];
        }
        for (int i = 0; i < vals.length; ++i) {
            vals[i] = vals[i] / sum;
        }
    }
    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            // handle exception
        }
        return bitmap;
    }
}
