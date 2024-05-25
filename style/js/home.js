// const logo = document.querySelectorAll("#logo path");

// for (let i = 0; i < logo.length; i++) {
//     console.log(`Letter ${i} length is ${logo[i].getTotalLength()}`);
// }


// icon loading
$(window).on("load", function () {
  $(".content").removeClass('hidden').addClass('fade-in').fadeIn(1000);
  $(".loader").fadeOut(3000, function () { 
  });
});

// scroll anima
ScrollReveal({
  // reset: true,
  distance: "80px",
  duration: 2000,
  // delay: 400,
});

ScrollReveal().reveal(".left", { delay: 250, origin: "left", interval: 200 });
ScrollReveal().reveal(".right", { delay: 250, origin: "right", interval: 150 });
ScrollReveal().reveal(".top", { delay: 250, origin: "top", interval: 200 });


// click btn home
document.getElementById('navigateButton').addEventListener('click', function() {
  window.location.href = '/index_start_code_run.html';
});

// nav scroll

window.addEventListener('scroll', function() {
  var navbar = document.getElementById('navbar');
  if (window.scrollY > 1) {
      navbar.classList.add('scrolled');
  } else {
      navbar.classList.remove('scrolled');
  }
});