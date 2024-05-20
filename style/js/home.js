// const logo = document.querySelectorAll("#logo path");

// for (let i = 0; i < logo.length; i++) {
//     console.log(`Letter ${i} length is ${logo[i].getTotalLength()}`);
// }


$(window).on('load', function() {
  const spinnerAnimationDuration = 500;

  // الكلاسات التي تريد إزالتها وإعادة إضافتها
  const classesToToggle = ['name', 'top'];

  // حدد كل عناصر المحتوى
  const contentElements = $(".content");

  // إزالة الكلاسات المحددة أثناء مرحلة اللودينج
  contentElements.each(function() {
    const $this = $(this);
    classesToToggle.forEach(function(cls) {
      $this.removeClass(cls);
    });
  });

  // Fade out the loader after spinnerAnimationDuration
  setTimeout(function() {
    $(".loader").fadeOut(1000, function() {
      // بعد اختفاء اللودر، أعد إضافة الكلاسات التي أزلتها وقم بتشغيل الكلاسات الخاصة بالأنيميشن
      contentElements.each(function() {
        const $this = $(this);

        // إعادة إضافة الكلاسات المحددة
        classesToToggle.forEach(function(cls) {
          $this.addClass(cls);
        });

        // إضافة الكلاس الخاص بالأنيميشن
        $this.removeClass('hidden').addClass('fade-in');
      });
    });
  }, spinnerAnimationDuration);
});


// scroll anima
ScrollReveal({
  // reset: true,
  distance: "80px",
  duration: 2000,
  delay: 400,
});

ScrollReveal().reveal(".name", { delay: 250, origin: "left", interval: 200 });
ScrollReveal().reveal(".right", { delay: 250, origin: "right", interval: 150 });
ScrollReveal().reveal(".top", { delay: 250, origin: "top", interval: 150 });

// add class if scroll
// $(window).on("load", function () {
//   $(".loadding-page").delay(6000).fadeOut(200);
//   $(".cssload-box-loading").on("click", function () {
//     $(".cssload-box-loading").fadeOut(200);
//   });
// });

// // script.js

// $(window).on('load', function() {
//   const spinnerAnimationDuration = 2000;
//   setTimeout(function() {
//     $(".loader").fadeOut(1000);
//     $(".content").removeClass('hidden').addClass('fade-in');
//   }, spinnerAnimationDuration);
// });

