Component({
  properties: {
    backTop: { // 属性名
      type: Boolean,
      value: ''
    }
  },
  methods: {
    itemclick: function (e) {
      wx.pageScrollTo({
        scrollTop: 0,
        duration: 400
      })
    }
  }
})
