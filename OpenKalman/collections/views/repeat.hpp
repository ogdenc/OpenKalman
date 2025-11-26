/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref collections::repeat_tuple_view and \ref collections::views::repeat.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_REPEAT_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_REPEAT_HPP

#include "values/values.hpp"
#include "all.hpp"

namespace OpenKalman::collections
{
  /**
   * \brief A \ref uniformly_gettable view that replicates a particular value N number of times
   * \tparam N The number of copies
   * \tparam T The type of the object to be replicated
   */
  template<std::size_t N, typename T>
  struct repeat_tuple_view
  {
#ifdef __cpp_lib_concepts
    constexpr repeat_tuple_view() requires std::default_initializable<T> = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::default_initializable<T>, int> = 0>
    constexpr repeat_tuple_view() {};
#endif


#ifdef __cpp_lib_concepts
    template<typename Arg> requires std::constructible_from<T, Arg&&>
#else
    template<typename Arg, std::enable_if_t<stdex::constructible_from<T, Arg&&>, int> = 0>
#endif
    explicit constexpr repeat_tuple_view(Arg&& arg) : t {std::forward<Arg>(arg)} {}


    /**
     * \return The underlying replicated value
     */
    constexpr T value() const { return t; }


    /**
     * \brief Get element i of a \ref repeat_tuple_view
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      static_assert(i < N, "Index out of range");
      return std::forward<decltype(self)>(self).t;
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() & { static_assert(i < N, "Index out of range"); return t; }

    /// \overload
    template<std::size_t i>
    constexpr decltype(auto)
    get() const & { static_assert(i < N, "Index out of range"); return t; }

    /// \overload
    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept { static_assert(i < N, "Index out of range"); return std::move(*this).t; }

    /// \overload
    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept { static_assert(i < N, "Index out of range"); return std::move(*this).t; }
#endif

  private:

    T t;
  };

}


namespace std
{
  template<std::size_t N, typename T>
  struct tuple_size<OpenKalman::collections::repeat_tuple_view<N, T>> : std::integral_constant<size_t, N> {};


  template<std::size_t i, std::size_t N, typename T>
  struct tuple_element<i, OpenKalman::collections::repeat_tuple_view<N, T>>
  {
    static_assert(i < N);
    using type = T;
  };
}


namespace OpenKalman::collections::views
{
  namespace detail
  {
    struct repeat_adaptor
    {
#ifdef __cpp_lib_ranges
      template<std::move_constructible W, values::size Bound = values::unbounded_size_t>
#else
      template<typename W, typename Bound = values::unbounded_size_t, typename = void>
#endif
      constexpr auto
      operator() [[nodiscard]] (W&& value, Bound&& bound = {}) const
      {
        if constexpr (stdex::same_as<Bound, values::unbounded_size_t>)
        {
          return stdex::ranges::views::repeat(std::forward<W>(value)) | all;
        }
        else if constexpr (values::fixed<Bound>)
        {
          if constexpr (values::fixed_value_of_v<Bound> == 1)
            return stdex::ranges::views::single(std::forward<W>(value)) | all;
          else
            return repeat_tuple_view<values::fixed_value_of_v<Bound>, W> {std::forward<W>(value)} | all;
        }
        else
        {
          return stdex::ranges::views::repeat(std::forward<W>(value), std::forward<Bound>(bound)) | all;
        }
      }

    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure for a set of repeated \ref collection objects.
   */
  inline constexpr detail::repeat_adaptor repeat;

}


#endif 
