package com.alxfabio.app_track
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent

import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.camera.core.CameraSelector

import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme

import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import androidx.compose.material3.Text as Text1

class MainActivity : ComponentActivity() {
    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        cameraExecutor = Executors.newSingleThreadExecutor()

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    CameraPreviewWithFaceDetection()
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}

@Composable
fun CameraPreviewWithFaceDetection() {
    val context = LocalContext.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val faceDetected = remember { mutableStateOf(false) }
    val userIsReal = remember { mutableStateOf(false) }
    val blinkDetected = remember { mutableStateOf(false) }
    val coroutineScope = rememberCoroutineScope()

    val previewView = remember { PreviewView(context) }

    val requestCameraPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (!isGranted) {
            Log.e("CameraX", "Camera permission denied")
        }
    }

    LaunchedEffect(Unit) {
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestCameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        } else {
            startCamera(context, previewView, faceDetected, userIsReal, blinkDetected, coroutineScope)
        }
    }

    Column {
        AndroidView(factory = { previewView }, modifier = Modifier.weight(1f))

        Text1(
            text = if (faceDetected.value) "Rostro detectado" else "No se detectó rostro",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(16.dp)
        )

        Text1(
            text = if (blinkDetected.value) "Parpadeo detectado" else "No se detectó parpadeo",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(16.dp)
        )

        Text1(
            text = if (userIsReal.value) "Usuario parece real" else "El usuario no parece real",
            style = MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(16.dp)
        )
    }
}

@OptIn(ExperimentalGetImage::class)
private fun startCamera(
    context: android.content.Context,
    previewView: PreviewView,
    faceDetected: MutableState<Boolean>,
    userIsReal: MutableState<Boolean>,
    blinkDetected: MutableState<Boolean>,
    coroutineScope: CoroutineScope
) {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

    cameraProviderFuture.addListener({
        val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

        val preview = Preview.Builder().build().also {
            it.setSurfaceProvider(previewView.surfaceProvider)
        }

        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

        val faceDetectorOptions = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .build()

        val faceDetector = FaceDetection.getClient(faceDetectorOptions)

        val imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also { imageAnalysis ->
                imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(context)) { imageProxy ->
                    processImageProxy(faceDetector, imageProxy, faceDetected, userIsReal, blinkDetected, coroutineScope)
                }
            }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                context as ComponentActivity,
                cameraSelector,
                preview,
                imageAnalyzer
            )
        } catch (exc: Exception) {
            Log.e("CameraX", "Use case binding failed", exc)
        }
    }, ContextCompat.getMainExecutor(context))
}

@androidx.camera.core.ExperimentalGetImage
private fun processImageProxy(
    faceDetector: com.google.mlkit.vision.face.FaceDetector,
    imageProxy: ImageProxy,
    faceDetected: MutableState<Boolean>,
    userIsReal: MutableState<Boolean>,
    blinkDetected: MutableState<Boolean>,
    coroutineScope: CoroutineScope
) {
    val mediaImage = imageProxy.image
    if (mediaImage != null) {
        val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

        faceDetector.process(image)
            .addOnSuccessListener { faces ->
                faceDetected.value = faces.isNotEmpty()
                if (faces.isNotEmpty()) {
                    val face = faces[0]
                    val leftEyeOpenProb = face.leftEyeOpenProbability ?: -1.0f
                    val rightEyeOpenProb = face.rightEyeOpenProbability ?: -1.0f

                    detectBlink(leftEyeOpenProb, rightEyeOpenProb, blinkDetected, userIsReal, coroutineScope)
                } else {
                    blinkDetected.value = false
                    userIsReal.value = false
                }
            }
            .addOnFailureListener {
                Log.e("FaceDetection", "Face detection failed", it)
                faceDetected.value = false
                blinkDetected.value = false
                userIsReal.value = false
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }
}

private fun detectBlink(
    leftEyeOpenProb: Float,
    rightEyeOpenProb: Float,
    blinkDetected: MutableState<Boolean>,
    userIsReal: MutableState<Boolean>,
    coroutineScope: CoroutineScope
) {
    val eyesOpen = leftEyeOpenProb > 0.8f && rightEyeOpenProb > 0.8f
    val eyesClosed = leftEyeOpenProb < 0.1f && rightEyeOpenProb < 0.1f

    if (eyesClosed && !blinkDetected.value) {
        blinkDetected.value = true
        coroutineScope.launch {
            delay(1000) // Espera 1 segundo
            if (eyesOpen) {
                userIsReal.value = true
            }
            delay(2000) // Espera 2 segundos más
            blinkDetected.value = false
        }
    }
}
//@Preview(showBackground = true)
//@Composable
//fun GreetingPreview() {
//    App_trackTheme {
//        Greeting("Android")
//    }
//}