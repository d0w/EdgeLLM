package logger

import (
	"log"

	"github.com/fatih/color"
)

// Logger represents a structured logger
type Logger struct {
	level string
}

// New creates a new logger with the specified level
func New(level string) *Logger {
	return &Logger{
		level: level,
	}
}

// Info logs an info message
func (l *Logger) Info(msg string, keysAndValues ...interface{}) {
	log.Printf("%s %s %v", color.HiBlueString("[INFO]"), msg, keysAndValues)
}

// Error logs an error message
func (l *Logger) Error(msg string, keysAndValues ...interface{}) {
	log.Printf("%s %s %v", color.HiRedString("[ERROR]"), msg, keysAndValues)
}

// Debug logs a debug message
func (l *Logger) Debug(msg string, keysAndValues ...interface{}) {
	if l.level == "debug" {
		log.Printf("%s %s %v", color.HiCyanString("[DEBUG]"), msg, keysAndValues)
	}
}

// Warn logs a warning message
func (l *Logger) Warn(msg string, keysAndValues ...interface{}) {
	log.Printf("%s %s %v", color.HiYellowString("[WARN]"), msg, keysAndValues)
}

// Fatal logs a fatal message and exits
func (l *Logger) Fatal(msg string, keysAndValues ...interface{}) {
	log.Printf("%s %s %v", color.HiMagentaString("[FATAL]"), msg, keysAndValues)
	// os.Exit(1)
}
