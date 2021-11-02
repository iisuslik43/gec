#!/usr/bin/env node


var fs = require('fs')
var dictionary = require('dictionary-en')
var nspell = require('nspell')


dictionary(ondictionary)


function ondictionary(err, dict) {
    if (err) {
        throw err
    }

    var spell = nspell(dict)

    fs.readFile('../data/example.txt', 'utf8', (err, data) => {
        if (err) {
            console.error(err)
            return
        }
        for (let line of data.split('\n')) {
            let res = ''
            for (let word of line.split(' ')) {
                if (spell.correct(word)) {
                    res += word + ' '
                } else {
                    res += word + '(' + spell.suggest(word)[0] + ') '
                }
            }
            console.log(res)
        }
    })

}