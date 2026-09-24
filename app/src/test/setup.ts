import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// jsdom has no layout, so MessageList's auto-scroll needs a stub.
Element.prototype.scrollIntoView = () => {}

afterEach(cleanup)
