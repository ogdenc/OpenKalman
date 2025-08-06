/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions relating to a compatible replacement for std::invoke.
 */

#ifndef OPENKALMAN_COMPATIBILITY_INVOKE_HPP
#define OPENKALMAN_COMPATIBILITY_INVOKE_HPP

namespace OpenKalman::stdcompat
{
#if __cplusplus >= 202002L
  using std::invoke;
#else
  namespace detail
  {
    template<typename> static constexpr bool is_reference_wrapper_v = false;
    template<typename U> static constexpr bool is_reference_wrapper_v<stdcompat::reference_wrapper<U>> = true;

    template<typename C, typename P, typename O, typename...Args>
    static constexpr decltype(auto)
    invoke_memptr(P C::* member, O&& object, Args&&... args)
    {
      if constexpr (std::is_function_v<P>)
      {
        if constexpr (stdcompat::same_as<C, remove_cvref_t<O>> or std::is_base_of_v<C, remove_cvref_t<O>>)
          return (std::forward<O>(object) .* member)(std::forward<Args>(args)...);
        else if constexpr (is_reference_wrapper_v<remove_cvref_t<O>>)
          return (object.get() .* member)(std::forward<Args>(args)...);
        else
          return ((*std::forward<O>(object)) .* member)(std::forward<Args>(args)...);
      }
      else
      {
        static_assert(std::is_object_v<P> && sizeof...(args) == 0);
        if constexpr (stdcompat::same_as<C, remove_cvref_t<O>> or std::is_base_of_v<C, remove_cvref_t<O>>)
          return std::forward<O>(object) .* member;
        else if constexpr (is_reference_wrapper_v<remove_cvref_t<O>>)
          return object.get() .* member;
        else
          return (*std::forward<O>(object)) .* member;
      }
    }
  }


  /**
   * \internal
   * \brief A constexpr version of std::invoke, for use when compiling in c++17
   **/
  template<typename F, typename...Args>
  static constexpr decltype(auto)
  invoke(F&& f, Args&&... args) noexcept(std::is_nothrow_invocable_v<F, Args...>)
  {
    if constexpr (std::is_member_pointer_v<remove_cvref_t<F>>)
      return invoke_memptr(f, std::forward<Args>(args)...);
    else
      return std::forward<F>(f)(std::forward<Args>(args)...);
  }
#endif


#if __cplusplus >= 202302L
  using std::invoke_r;
#else
  /**
   * \internal
   * \brief A constexpr version of std::invoke_r, for use when compiling in c++17
   **/
#ifdef __cpp_concepts
  template<typename R, typename F, typename...Args> requires std::is_invocable_r_v<R, F, Args...>
#else
  template<typename R, typename F, typename...Args, std::enable_if_t<std::is_invocable_r_v<R, F, Args...>, int> = 0>
#endif
  constexpr R invoke_r(F&& f, Args&&... args) noexcept(std::is_nothrow_invocable_r_v<R, F, Args...>)
  {
    if constexpr (std::is_void_v<R>)
      stdcompat::invoke(std::forward<F>(f), std::forward<Args>(args)...);
    else
      return stdcompat::invoke(std::forward<F>(f), std::forward<Args>(args)...);
  }
#endif

}


#endif