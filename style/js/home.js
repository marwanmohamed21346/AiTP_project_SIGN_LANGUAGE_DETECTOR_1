// const logo = document.querySelectorAll("#svg-about path");

// for (let i = 0; i < logo.length; i++) {
//     console.log(`Letter ${i} length is ${logo[i].getTotalLength()}`);
// }


// scroll trans
ScrollReveal({ 
  // reset: true,
  distance: '60px',
  duration: 2000,
  delay: 400
});

ScrollReveal().reveal('.name', { delay: 250, origin: 'left', interval: 200});
ScrollReveal().reveal('.right', { delay: 250, origin: 'right', interval: 150});
ScrollReveal().reveal('.top', { delay: 250, origin: 'top', interval: 150});





const logo = document.querySelectorAll("#logo path");

for (let i = 0; i < logo.length; i++) {
    console.log(`Letter ${i} length is ${logo[i].getTotalLength()}`);
}
