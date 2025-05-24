library(shiny)
library(httr)
library(jsonlite)

backend_url <- "http://localhost:8000/predict"  # Change if backend runs elsewhere
# If backend is deployed elsewhere, update this URL accordingly.

ui <- fluidPage(
  titlePanel("Scleroderma Prediction & Test Recommendation"),
  sidebarLayout(
    sidebarPanel(
      fileInput("nlpfile", "Upload Doctor's Notes (txt)", accept = ".txt"),
      textAreaInput("nlptext", "Or Paste Doctor's Notes / Symptoms", height = "150px"),
      actionButton("predict_btn", "Predict Scleroderma & Recommend Tests", class = "btn-primary")
    ),
    mainPanel(
      h4("Prediction Result"),
      verbatimTextOutput("prediction"),
      h4("Recommended Next Tests"),
      uiOutput("tests")
    )
  )
)

server <- function(input, output, session) {
  observeEvent(input$predict_btn, {
    # Get text from file or textarea
    text <- NULL
    if (!is.null(input$nlpfile)) {
      text <- paste(readLines(input$nlpfile$datapath, warn = FALSE), collapse = " ")
    } else if (nzchar(input$nlptext)) {
      text <- input$nlptext
    }
    if (is.null(text) || text == "") {
      output$prediction <- renderText("Please upload a file or enter text.")
      output$tests <- renderUI(HTML(""))
      return()
    }
    # Send to backend
    res <- tryCatch({
      httr::POST(
        backend_url,
        body = list(text = text),
        encode = "json",
        timeout(10)
      )
    }, error = function(e) NULL)
    if (is.null(res) || httr::http_error(res)) {
      output$prediction <- renderText("Error: Could not connect to backend API.")
      output$tests <- renderUI(HTML(""))
      return()
    }
    result <- content(res, as = "parsed", simplifyVector = TRUE)
    output$prediction <- renderText({
      paste0(
        "Prediction: ", result$prediction, "\n",
        "Probability of Scleroderma: ", round(result$scleroderma_probability * 100, 1), "%"
      )
    })
    output$tests <- renderUI({
      if (!is.null(result$recommended_tests) && length(result$recommended_tests) > 0) {
        HTML(paste("<ul>", paste(paste0("<li>", result$recommended_tests, "</li>"), collapse = ""), "</ul>"))
      } else {
        HTML("No additional tests recommended.")
      }
    })
  })
}

shinyApp(ui, server)
