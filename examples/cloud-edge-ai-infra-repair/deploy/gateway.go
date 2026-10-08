// Copyright 2026 The KubeEdge Authors.
// SPDX-License-Identifier: Apache-2.0
// The edge serving gateway is the same release for both CPU architectures.
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

type config struct {
	Revision string `json:"revision"`
	Backend  string `json:"backend_url"`
}

type flushingWriter struct{ w http.ResponseWriter }

func (f flushingWriter) Write(p []byte) (int, error) {
	n, err := f.w.Write(p)
	if flusher, ok := f.w.(http.Flusher); ok {
		flusher.Flush()
	}
	return n, err
}

func main() {
	client := &http.Client{Timeout: 120 * time.Second,
		Transport:     &http.Transport{Proxy: nil},
		CheckRedirect: func(req *http.Request, via []*http.Request) error { return http.ErrUseLastResponse }}
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		data, err := os.ReadFile("/config/service.json")
		var cfg config
		if err != nil || json.Unmarshal(data, &cfg) != nil {
			http.Error(w, "serving configuration unavailable", http.StatusServiceUnavailable)
			return
		}
		if !strings.HasPrefix(cfg.Backend, "http://") || (cfg.Revision != "R1" && cfg.Revision != "R2") {
			http.Error(w, "invalid serving configuration", http.StatusServiceUnavailable)
			return
		}
		if r.URL.Path == "/health" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{"revision": cfg.Revision, "backend_url": cfg.Backend})
			return
		}
		if r.Method != "POST" || (r.URL.Path != "/generate" && r.URL.Path != "/generate-stream") {
			http.NotFound(w, r)
			return
		}
		if r.URL.Path == "/generate-stream" && cfg.Revision != "R2" {
			http.Error(w, "R1 does not support the required streaming protocol", http.StatusServiceUnavailable)
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 8192))
		if err != nil {
			http.Error(w, "request too large", http.StatusBadRequest)
			return
		}
		request, err := http.NewRequest("POST", cfg.Backend+r.URL.Path, strings.NewReader(string(body)))
		if err != nil {
			http.Error(w, "invalid worker address", http.StatusBadGateway)
			return
		}
		request.Header.Set("Content-Type", "application/json")
		response, err := client.Do(request)
		if err != nil {
			http.Error(w, "inference worker unavailable", http.StatusBadGateway)
			return
		}
		defer response.Body.Close()
		w.Header().Set("Content-Type", response.Header.Get("Content-Type"))
		w.Header().Set("X-Serving-Revision", cfg.Revision)
		w.WriteHeader(response.StatusCode)
		if _, err := io.Copy(flushingWriter{w}, response.Body); err != nil {
			log.Printf("stream interrupted: %v", err)
		}
	})
	fmt.Println("Edge inference gateway listening on 18080")
	log.Fatal(http.ListenAndServe(":18080", nil))
}
