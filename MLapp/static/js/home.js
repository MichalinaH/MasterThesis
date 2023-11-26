const input = document.getElementById("id_image")
const imageBox = document.getElementById("image_box")

input.addEventListener("change", ()=>{
    const img_data = input.files[0]
      const url = URL.createObjectURL(img_data)
      imageBox.innerHTML = `<img src="${url}" id="image" width=638px  height=500px >`
    
  })
