package main

import (
	"fmt"
	"net/http"
	"os"
	"sync"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

type User struct {
	Name string `json:"name"`
}

type App struct {
	config *oauth2.Config
}

var userCache = make(map[int]User)

var cacheMutex sync.RWMutex

func main() {
	// environment variables from google credentials (dont have them yet)
	clientid := os.Getenv("CLIENT_ID")
	clientSecret := os.Getenv("CLIENT_SECRET")

	conf := &oauth2.Config{
		ClientID:     clientid,
		ClientSecret: clientSecret,
		RedirectURL:  "", // empty for now but its just the url redirecting the login page of the mobile app
		Scopes:       []string{"email", "profile"},
		Endpoint:     google.Endpoint,
	}
	app := &App{config: conf}
	mux := http.NewServeMux()
	mux.HandleFunc("/", handleRoot)

	mux.HandleFunc("POST /users", createUser)

	// oauth
	mux.HandleFunc("GET /auth/login", app.loginHandler)
	mux.HandleFunc("GET /auth/oauth", app.oAuthHandler)
	mux.HandleFunc("GET /auth/callback", app.oAuthCallbackHandler)

	fmt.Println("Server Listening to : 8080")
	http.ListenAndServe(":8080", mux)
}
