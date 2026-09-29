import { useContext } from 'react'
import AppContext from './AppContextInstance'

export const useApp = () => useContext(AppContext)
