package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"text/template"

	"golang.org/x/oauth2"
)

func handleRoot(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello World")
}

func createUser(w http.ResponseWriter, r *http.Request) {
	var user User
	err := json.NewDecoder(r.Body).Decode(&user)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if user.Name == "" {
		http.Error(w, "name is required", http.StatusBadRequest)
		return
	}

	cacheMutex.Lock()
	userCache[len(userCache)+1] = user
	cacheMutex.Unlock()

	w.WriteHeader(http.StatusNoContent)
}

// login handler
func (a *App) loginHandler(w http.ResponseWriter, r *http.Request) {
	t, err := template.ParseFiles("index.html") //replace this file when we do the mobile app ui
	if err != nil {
		log.Print(err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	t.Execute(w, nil)
}

// oauth handler, redirect the user to the oatuh provider
func (a *App) oAuthHandler(w http.ResponseWriter, r *http.Request) {
	url := a.config.AuthCodeURL("state", oauth2.AccessTypeOffline)
	http.Redirect(w, r, url, http.StatusTemporaryRedirect)
}

func (a *App) oAuthCallbackHandler(w http.ResponseWriter, r *http.Request) {
	code := r.URL.Query().Get("code")

	t, err := a.config.Exchange(context.Background(), code)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// save the token t in some persistent storage here

	// Creating an HTTP client to make authenticated request using the access key.
	// This client method also regenerate the access key using the refresh key.
	client := a.config.Client(context.Background(), t)

	// Getting the user public details from google API endpoint
	resp, err := client.Get("https://www.googleapis.com/oauth2/v2/userinfo")
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Closing the request body when this function returns.
	// This is a good practice to avoid memory leak
	defer resp.Body.Close()

	var v any

	// Reading the JSON body using JSON decoder
	err = json.NewDecoder(resp.Body).Decode(&v)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// sending the user public value as a response. This is may not be a good practice,
	// but for demonstration, I think it serves the need.
	fmt.Fprintf(w, "%v", v)
}
