package edu.imac.nutc.tensorflowtest;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by cheng on 2018/2/19.
 */

public class InceptionV3Activity extends AppCompatActivity {
    private static final String TAG = InceptionV3Activity.class.getName();
    private static final String MODEL_FILE = "file:///android_asset/inception_v3_2016_08_28_frozen.pb";
    private static final String LABEL_PATH = "imagenet_slim_labels.txt";
    private static final String INPUT_NODE = "input";
    private static final String OUTPUT_NODE = "InceptionV3/Predictions/Reshape_1";
    private static int INPUT_SIZE = 299;
    private TensorFlowInferenceInterface inferenceInterface;
    private int[] intValues;
    private float[] inputFloat;
    private ArrayList<RecognitionUnit> result;
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_inception);
        init();
    }

    private void init() {
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILE);
        intValues = new int[INPUT_SIZE * INPUT_SIZE];
        inputFloat = new float[INPUT_SIZE * INPUT_SIZE * 3];
        result=new ArrayList<>();
        loadLabelModel();
        convertBitmapToByteBuffer(getBitmapFromAsset(this, "cat.jpeg"));
    }
    private void loadLabelModel(){
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(getAssets().open(LABEL_PATH)));
            String mLine;
            while ((mLine = reader.readLine()) != null) {
                RecognitionUnit recognitionUnit =new RecognitionUnit();
                recognitionUnit.setLabel(mLine);
                result.add(recognitionUnit);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
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
    }

    @Override
    protected void onPostResume() {
        super.onPostResume();

        inferenceInterface.feed(INPUT_NODE, inputFloat, 1, INPUT_SIZE, INPUT_SIZE, 3);
        final float[] resu = new float[1001];
        inferenceInterface.run(new String[]{OUTPUT_NODE});
        inferenceInterface.fetch(OUTPUT_NODE, resu);
        for(int i=0;i<1001;i++){
           result.get(i).setValue(resu[i]);
        }
        sortArray();
        for (int i=0;i<result.size();i++){
            Log.d(TAG,"label : "+result.get(i).getLabel()+"value : "+result.get(i).getValue());
        }
    }

    private void sortArray(){
        for(int i=0;i<result.size();i++){
            for(int j=i;j<result.size();j++){
                if(result.get(j).getValue()>result.get(i).getValue()){
                    String label=result.get(i).getLabel();
                    float value=result.get(i).getValue();
                    result.get(i).setLabel(result.get(j).getLabel());
                    result.get(i).setValue(result.get(j).getValue());
                    result.get(j).setLabel(label);
                    result.get(j).setValue(value);
                }
            }
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
