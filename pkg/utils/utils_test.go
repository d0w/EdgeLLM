package utils

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStringInSlice(t *testing.T) {
	slice := []string{"apple", "banana", "cherry"}
	
	assert.True(t, StringInSlice("apple", slice))
	assert.True(t, StringInSlice("banana", slice))
	assert.False(t, StringInSlice("orange", slice))
	assert.False(t, StringInSlice("", slice))
}

func TestToJSON(t *testing.T) {
	data := map[string]interface{}{
		"name": "test",
		"age":  25,
	}
	
	jsonStr, err := ToJSON(data)
	assert.NoError(t, err)
	assert.Contains(t, jsonStr, "test")
	assert.Contains(t, jsonStr, "25")
}

func TestFromJSON(t *testing.T) {
	jsonStr := `{"name":"test","age":25}`
	var data map[string]interface{}
	
	err := FromJSON(jsonStr, &data)
	assert.NoError(t, err)
	assert.Equal(t, "test", data["name"])
	assert.Equal(t, float64(25), data["age"]) // JSON numbers are float64
}

func TestTrimAndLower(t *testing.T) {
	assert.Equal(t, "hello", TrimAndLower("  HELLO  "))
	assert.Equal(t, "world", TrimAndLower("World"))
	assert.Equal(t, "", TrimAndLower("   "))
}

func TestFormatError(t *testing.T) {
	err := errors.New("original error")
	formatted := FormatError(err, "test context")
	
	assert.Error(t, formatted)
	assert.Contains(t, formatted.Error(), "test context")
	assert.Contains(t, formatted.Error(), "original error")
	
	// Test with nil error
	nilFormatted := FormatError(nil, "test context")
	assert.NoError(t, nilFormatted)
} 