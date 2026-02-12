package com.example.acc_gyrodata

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat

class MainActivity : AppCompatActivity(), SensorStreamer.StreamListener {

    private lateinit var etIpAddress: EditText
    private lateinit var etPort: EditText
    private lateinit var btnConnect: Button
    private lateinit var btnToggleStream: Button
    private lateinit var tvStatus: TextView

    private lateinit var sensorStreamer: SensorStreamer
    private var isStreaming = false
    private var isConnected = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        etIpAddress = findViewById(R.id.etIpAddress)
        etPort = findViewById(R.id.etPort)
        btnConnect = findViewById(R.id.btnConnect)
        btnToggleStream = findViewById(R.id.btnToggleStream)
        tvStatus = findViewById(R.id.tvStatus)

        sensorStreamer = SensorStreamer(this, this)

        btnConnect.setOnClickListener {
            if (!isConnected) {
                val ip = etIpAddress.text.toString().trim()
                val portStr = etPort.text.toString().trim()
                if (ip.isNotEmpty() && portStr.isNotEmpty()) {
                    val port = portStr.toInt()
                    updateStatus("Connecting...", R.color.text_secondary)
                    sensorStreamer.connect(ip, port)
                } else {
                    Toast.makeText(this, "Please enter IP and Port", Toast.LENGTH_SHORT).show()
                }
            } else {
                sensorStreamer.disconnect()
                onConnectionStatusChanged(false, "Disconnected")
            }
        }

        btnToggleStream.setOnClickListener {
            if (isStreaming) {
                sensorStreamer.stopStreaming()
                btnToggleStream.text = "Start Streaming"
                isStreaming = false
            } else {
                sensorStreamer.startStreaming()
                btnToggleStream.text = "Stop Streaming"
                isStreaming = true
            }
        }
    }

    override fun onConnectionStatusChanged(isConnected: Boolean, message: String) {
        runOnUiThread {
            this.isConnected = isConnected
            val colorRes = if (isConnected) R.color.status_connected else R.color.status_disconnected
            updateStatus(message, colorRes)
            btnConnect.text = if (isConnected) "Disconnect" else "Connect"
            btnToggleStream.isEnabled = isConnected

            if (!isConnected) {
                isStreaming = false
                btnToggleStream.text = "Start Streaming"
            }
        }
    }

    override fun onError(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        }
    }

    private fun updateStatus(text: String, colorRes: Int) {
        tvStatus.text = text
        tvStatus.setTextColor(ContextCompat.getColor(this, colorRes))
    }

    override fun onPause() {
        super.onPause()
        if (isStreaming) {
            sensorStreamer.stopStreaming()
            btnToggleStream.text = "Start Streaming"
            isStreaming = false
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorStreamer.disconnect()
    }
}
