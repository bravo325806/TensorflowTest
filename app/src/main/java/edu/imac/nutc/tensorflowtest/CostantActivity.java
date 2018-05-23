package edu.imac.nutc.tensorflowtest;

import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.TensorFlow;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

/**
 * Created by cheng on 2018/1/23.
 */

public class CostantActivity extends AppCompatActivity implements View.OnClickListener{
    private EditText editTextOne;
    private EditText editTextTwo;
    private EditText editTextThree;
    private TextView textOne;
    private TextView textTwo;
    private static final String MODEL_FILE = "file:///android_asset/frozen_V3.pb";
    private static final String INPUT_NODE = "input_1";
    private static final String OUTPUT_NODE = "predictions/Softmax";
    private TensorFlowInferenceInterface inferenceInterface;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_constant);
        init();
    }
    private void init(){
        editTextOne=findViewById(R.id.edittext_one);
        editTextTwo=findViewById(R.id.edittext_two);
        editTextThree=findViewById(R.id.edittext_three);
        textOne=findViewById(R.id.text1);
        textTwo=findViewById(R.id.text2);
        findViewById(R.id.button).setOnClickListener(this);
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(),MODEL_FILE);
    }

    @Override
    public void onClick(View view) {
        if (editTextOne.getText().toString().equals("")||editTextTwo.getText().toString().equals("")||editTextThree.getText().toString().equals("")){
            return;
        }
        float num1 =Float.valueOf(editTextOne.getText().toString());
        float num2 =Float.valueOf(editTextTwo.getText().toString());
        float num3 =Float.valueOf(editTextThree.getText().toString());
        float[] inputFloats = {num1,num2,num3};
        inferenceInterface.feed(INPUT_NODE,inputFloats,1,1,1,3);
        boolean bool[]={false};
        inferenceInterface.feed("batch_normalization_1/keras_learning_phase",bool,1);
        float[] resu = new float[1];
        inferenceInterface.run(new String[]{OUTPUT_NODE});
        inferenceInterface.fetch(OUTPUT_NODE,resu);
        textOne.setText(String.valueOf(resu[0]));
//        textTwo.setText(String.valueOf(resu[1]));
    }
}
