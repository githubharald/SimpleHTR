/*!
* Start Bootstrap - Scrolling Nav v5.0.6 (https://startbootstrap.com/template/scrolling-nav)
* Copyright 2013-2023 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-scrolling-nav/blob/master/LICENSE)
*/
//
// Scripts
// 

window.addEventListener('DOMContentLoaded', event => {

  // Activate Bootstrap scrollspy on the main nav element
  const mainNav = document.body.querySelector('#mainNav');
  if (mainNav) {
      new bootstrap.ScrollSpy(document.body, {
          target: '#mainNav',
          rootMargin: '0px 0px -40%',
      });
  };

  // Collapse responsive navbar when toggler is visible
  const navbarToggler = document.body.querySelector('.navbar-toggler');
  const responsiveNavItems = [].slice.call(
      document.querySelectorAll('#navbarResponsive .nav-link')
  );
  responsiveNavItems.map(function (responsiveNavItem) {
      responsiveNavItem.addEventListener('click', () => {
          if (window.getComputedStyle(navbarToggler).display !== 'none') {
              navbarToggler.click();
          }
      });
  });



  const fileInput = document.getElementById('fileInput');
  // Get the dropzone element
  const dropzone = document.getElementById('dropzone');

  // Function to handle file selection when a file is dropped on the dropzone
  function handleDrop(event) {
    event.preventDefault();
    dropzone.style.border = "dashed 2px #ccc"; // Reset border style
    const files = event.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
  }

  // Function to display the selected file name
  function displayFile(file) {
    const fileName = file.name;
    console.log(`Selected file: ${fileName}`);
  }

  // Function to handle file selection when the button is clicked
  function handleFileSelect() {
    fileInput.click();
  }

  // Function to handle file selection when a file is chosen via the file input
  fileInput.addEventListener('change', () => {
    const selectedFile = fileInput.files[0];
    if (selectedFile) {
        uploadFile(selectedFile);
    }
  });

  function handleResponse(responseText) {
    // Assuming the server returns the content of "transcribed.txt"
    // Update the content of the "outputBox" div with the transcribed text
    document.getElementById('outputBox').textContent = responseText;
    const audioPlayer = document.getElementById('audioPlayer');
    
    const playButton = document.getElementById('playButton');
    audio_file_path = './output.mp3'

    audioPlayer.src = audio_file_path
    console.log('Audio Player src:', audioPlayer.src);
    // Show the audio player and "Play Audio" button
    audioPlayer.style.display = 'block';
    playButton.style.display = 'block';
  }

  // Function to upload the selected file to the Flask backend
  function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload', true);

    xhr.onreadystatechange = function () {
      if (xhr.readyState === XMLHttpRequest.DONE) {
        if (xhr.status === 200) {
          // Handle the response from the server
          handleResponse(xhr.responseText);
        } else {
          // Handle error responses from the server
          console.error('Error:', xhr.responseText);
        }
      }
    };

    xhr.send(formData);
  }

  // Prevent default behavior for drag events to enable dropping files
  dropzone.addEventListener('dragenter', (event) => {
    event.preventDefault();
    dropzone.style.border = "dashed 2px #0080ff"; // Show a visual indicator
  });

  dropzone.addEventListener('dragleave', (event) => {
    event.preventDefault();
    dropzone.style.border = "dashed 2px #ccc"; // Reset border style
  });

  dropzone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropzone.style.border = "dashed 2px #0080ff"; // Show a visual indicator
  });

  dropzone.addEventListener('drop', handleDrop);

  const playButton = document.getElementById('playButton');
    playButton.addEventListener('click', () => {
      const audioPlayer = document.getElementById('audioPlayer');
      audioPlayer.play();
    });

});