// const logo = document.querySelectorAll("#logo path");

// for (let i = 0; i < logo.length; i++) {
//     console.log(`Letter ${i} length is ${logo[i].getTotalLength()}`);
// }

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

// slid show

var swiper = new Swiper(".mySwiper", {
  slidesPerView: 3,
  spaceBetween: 30,
  pagination: {
    el: ".swiper-pagination",
    clickable: true,
  },
  breakpoints: {
    310: {
      slidesPerView: 1,
    },
    768: {
      slidesPerView: 2,
    },
    992: {
      slidesPerView: 3,
    },
  },
});
