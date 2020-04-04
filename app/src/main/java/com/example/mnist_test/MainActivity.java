package com.example.mnist_test;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.os.Build;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.w3c.dom.Text;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;


public class MainActivity extends AppCompatActivity {

    ImageView ivDigit;
    Button btnChoose;
    Button btnPredict;
    TextView tvDebug;

    Interpreter tflite;
    Interpreter.Options options = new Interpreter.Options();
    ByteBuffer imgData = null;
    int[] imgPixels = new int[28*28];
    float[][] result = new float[1][10];

    private static final int IMAGE_PICK_CODE = 1000;
    private static final int PERMISSION_CODE = 1001;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        ivDigit = findViewById(R.id.ivImage);
        btnChoose = findViewById(R.id.btnChoose);
        btnPredict = findViewById(R.id.btnPredict);
        tvDebug = findViewById(R.id.tvDebug);

        try {
            tflite = new Interpreter(loadModelFile(), options);
        }catch (Exception e){
            e.printStackTrace();
        }

        imgData = ByteBuffer.allocateDirect(4*28*28);
        imgData.order(ByteOrder.nativeOrder());

        // choose image
        btnChoose.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(Build.VERSION.SDK_INT == Build.VERSION_CODES.M){
                    if(checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) ==
                            PackageManager.PERMISSION_DENIED){
                        String[] permissions = {Manifest.permission.READ_EXTERNAL_STORAGE};
                        requestPermissions(permissions, PERMISSION_CODE);
                    }else{
                        chooseImageFromGallery();
                    }
                }else{
                    chooseImageFromGallery();
                }
            }
        });

        // predict
        btnPredict.setOnClickListener(new View.OnClickListener() {
            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View v) {
                ivDigit.invalidate();
                BitmapDrawable drawable = (BitmapDrawable) ivDigit.getDrawable();
                Bitmap bitmap = drawable.getBitmap();
                Bitmap bitmap_resize = getResizedBitmap(bitmap, 28, 28);
                convertBitmapToByteBuffer(bitmap_resize);

                tflite.run(imgData, result);

                tvDebug.setText("result= " + argmax(result[0]));
            }
        });




    }

    private int argmax(float[] probs){
        int maxIds = -1;

        float maxProb = 0.0f;

        for (int i=0; i<probs.length; i++){
            if(probs[i] > maxProb){
                maxProb = probs[i];
                maxIds = i;
            }
        }

        return maxIds;
    }


    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();

        bitmap.getPixels(imgPixels, 0, bitmap.getWidth(), 0, 0,
                bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                int value = imgPixels[pixel++];
                imgData.putFloat(convertPixel(value));
            }
        }
    }

    private static float convertPixel(int color) {
        return (255 - (((color >> 16) & 0xFF) * 0.299f
                + ((color >> 8) & 0xFF) * 0.587f
                + (color & 0xFF) * 0.114f)) / 255.0f;
    }


    private Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight){
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float)newWidth)/width;
        float scaleHeight = ((float)newHeight)/height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);

        return Bitmap.createBitmap(bm, 0,0,width, height, matrix, false);
    }

    private void chooseImageFromGallery(){
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType("image/*");
        startActivityForResult(intent, IMAGE_PICK_CODE);
    }


    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("mnist_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && requestCode == IMAGE_PICK_CODE) {
            assert data != null;
            ivDigit.setImageURI(data.getData());
        }
    }

    // runtime
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_CODE) {
            chooseImageFromGallery();
        }
    }


}
