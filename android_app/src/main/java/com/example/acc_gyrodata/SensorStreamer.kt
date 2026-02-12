package com.example.acc_gyrodata

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import java.io.BufferedWriter
import java.io.OutputStreamWriter
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.Executors

class SensorStreamer(
    context: Context,
    private val listener: StreamListener
) : SensorEventListener {

    interface StreamListener {
        fun onConnectionStatusChanged(isConnected: Boolean, message: String)
        fun onError(message: String)
    }

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

    private var socket: Socket? = null
    private var writer: BufferedWriter? = null
    private val executor = Executors.newSingleThreadExecutor()

    private var isStreaming = false

    private var lastAcc = FloatArray(3)

    fun connect(host: String, port: Int) {
        executor.execute {
            try {
                socket = Socket()
                socket?.connect(InetSocketAddress(host, port), 5000)
                writer = BufferedWriter(OutputStreamWriter(socket?.getOutputStream()))
                listener.onConnectionStatusChanged(true, "Connected to $host:$port")
            } catch (e: Exception) {
                listener.onConnectionStatusChanged(false, "Connection failed: ${e.message}")
            }
        }
    }

    fun startStreaming() {
        if (socket == null || socket?.isConnected == false) {
            listener.onError("Not connected to any host")
            return
        }
        isStreaming = true
        registerSensors()
    }

    fun stopStreaming() {
        isStreaming = false
        unregisterSensors()
    }

    private fun registerSensors() {
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        if (accelerometer == null) {
            listener.onError("Accelerometer sensor not available")
        }
    }

    private fun unregisterSensors() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (!isStreaming || event == null) return

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            System.arraycopy(event.values, 0, lastAcc, 0, 3)
            val timestampMs = event.timestamp / 1_000_000
            sendData(timestampMs)
        }
    }

    private fun sendData(timestampMs: Long) {
        val data = "$timestampMs,${lastAcc[0]},${lastAcc[1]},${lastAcc[2]}\n"

        executor.execute {
            try {
                writer?.write(data)
                writer?.flush()
            } catch (e: Exception) {
                isStreaming = false
                unregisterSensors()
                listener.onConnectionStatusChanged(false, "Connection lost: ${e.message}")
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    fun disconnect() {
        stopStreaming()
        executor.execute {
            try {
                writer?.close()
                socket?.close()
            } catch (e: Exception) {
                // Ignore
            } finally {
                socket = null
                writer = null
            }
        }
    }
}
