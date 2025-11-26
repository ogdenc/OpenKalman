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
 * \brief Definition for \ref collections::from_range.
 */

#ifndef OPENKALMAN_COLLECTIONS_VIEWS_FROM_RANGE_HPP
#define OPENKALMAN_COLLECTIONS_VIEWS_FROM_RANGE_HPP

#include <variant>
#include "values/values.hpp"
#include "collections/functions/get_size.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/functions/get.hpp"
#include "internal/movable_wrapper.hpp"

namespace OpenKalman::collections
{
  namespace detail_from_range
  {
#ifdef __cpp_concepts
    template<std::size_t i, typename T>
#else
    template<std::size_t i, typename T, typename = void>
#endif
    struct tuple_element_impl
    {
      using type = stdex::ranges::range_value_t<T>;

      template<typename R> constexpr decltype(auto) operator() (R&& r) const
       { return get<i>(std::forward<R>(r)); }
    };


#ifdef __cpp_concepts
    template<std::size_t i, typename T> requires gettable<i, T>
    struct tuple_element_impl<i, T>
#else
    template<std::size_t i, typename T>
    struct tuple_element_impl<i, T, std::enable_if_t<gettable<i, T>>>
#endif
      : collection_element<i, T>
    {
      template<typename R> constexpr decltype(auto) operator() (R&& r) const
       { return get<i>(std::forward<R>(r)); }
    };


#ifdef __cpp_concepts
    template<std::size_t i, typename T>
#else
    template<std::size_t i, typename T, typename = void>
#endif
    struct get_elem : tuple_element_impl<i, T> {};


    template<std::size_t i, typename R>
    inline constexpr decltype(auto)
    get_from_base(R&& r) { return get_elem<i, stdex::remove_cvref_t<R>>{}(std::forward<R>(r)); }


    template<typename T>
    struct get_elem<0, stdex::ranges::single_view<T>>
    {
      using type = T;
      template<typename U> constexpr decltype(auto) operator() (U&& u) const { return *std::forward<U>(u).data(); }
    };


    template<std::size_t i, typename R>
    struct get_elem<i, stdex::ranges::ref_view<R>> : get_elem<i, stdex::remove_cvref_t<R>>
    {
      template<typename T> constexpr decltype(auto) operator() (T&& t) const { return get_from_base<i>(std::forward<T>(t).base()); }
    };


    template<std::size_t i, typename R>
    struct get_elem<i, stdex::ranges::owning_view<R>> : get_elem<i, stdex::remove_cvref_t<R>>
    {
      template<typename T> constexpr decltype(auto) operator() (T&& t) const { return get_from_base<i>(std::forward<T>(t).base()); }
    };


#ifdef __cpp_concepts
    template<std::size_t i, typename V> requires (size_of_v<V> != stdex::dynamic_extent)
    struct get_elem<i, stdex::ranges::reverse_view<V>>
#else
    template<std::size_t i, typename V>
    struct get_elem<i, stdex::ranges::reverse_view<V>, std::enable_if_t<size_of_v<V> != stdex::dynamic_extent>>
#endif
      : get_elem<size_of_v<V> - i - 1_uz, stdex::remove_cvref_t<V>>
    {
      template<typename T> constexpr decltype(auto) operator() (T&& t) const
      {
        if constexpr (not std::is_lvalue_reference_v<T> or stdex::copy_constructible<V>)
          return get_from_base<size_of_v<V> - i - 1_uz>(std::forward<T>(t).base());
        else
          return get<i>(std::forward<T>(t));
      }
    };

  }


  /**
   * \brief A \ref collection_view created from a std::ranges::random_access_range that is a std::ranges::viewable_range.
   */
#ifdef __cpp_concepts
  template<stdex::ranges::random_access_range V> requires stdex::ranges::viewable_range<V>
#else
  template<typename V>
#endif
  struct from_range : stdex::ranges::view_interface<from_range<V>>
  {
  private:

    using RangeBox = internal::movable_wrapper<V>;

  public:

    /**
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    constexpr
    from_range() = default;
#else
    template<bool Enable = true, std::enable_if_t<Enable and stdex::default_initializable<RangeBox>, int> = 0>
    constexpr
    from_range() {}
#endif


    /**
     * \brief Construct from a \ref std::ranges::random_access_range.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires (not std::same_as<stdex::remove_cvref_t<Arg>, from_range>)
#else
    template<typename Arg, std::enable_if_t<(not std::is_same_v<stdex::remove_cvref_t<Arg>, from_range>), int> = 0>
#endif
    constexpr explicit 
    from_range(Arg&& arg) noexcept : r_ {std::forward<Arg>(arg)} {}


    /**
     * \brief The base tuple.
     **/
#ifdef __cpp_explicit_this_parameter
    constexpr decltype(auto)
    base(this auto&& self) noexcept { return std::forward<decltype(self)>(self).r_.get(); }
#else
    constexpr V& base() & { return this->r_.get(); }
    constexpr const V& base() const & { return this->r_.get(); }
    constexpr V&& base() && noexcept { return std::move(*this).r_.get(); }
    constexpr const V&& base() const && noexcept { return std::move(*this).r_.get(); }
#endif


    /**
     * \brief An iterator to the beginning of the range.
     */
    constexpr auto begin() { return stdex::ranges::begin(base()); }

    /// \overload
    constexpr auto begin() const { return stdex::ranges::begin(base()); }


    /**
     * \brief An iterator to the end of the range.
     */
    constexpr auto end() { return stdex::ranges::end(base()); }

    /// \overload
    constexpr auto end() const { return stdex::ranges::end(base()); }


    /**
     * \brief The size of the resulting object.
     */
#ifdef __cpp_concepts
    constexpr auto
    size() const requires sized<V>
#else
    template<bool Enable = true, std::enable_if_t<Enable and sized<V>, int> = 0>
    constexpr auto
    size() const
#endif
    {
      if constexpr (values::fixed_value_compares_with<size_of<V>, stdex::dynamic_extent>)
        return get_size(base());
      else
        return size_of<V>{};
    }


    /**
     * \returns Indicates whether the base object is empty.
     */
#ifdef __cpp_concepts
    constexpr auto
    empty() const requires sized<V> or stdex::ranges::forward_range<V>
#else
    template<bool Enable = true, std::enable_if_t<Enable and (sized<V> or stdex::ranges::forward_range<V>), int> = 0>
    constexpr auto
    empty() const
#endif
    {
      if constexpr (sized<V>)
      {
        if constexpr (size_of_v<V> == stdex::dynamic_extent) return get_size(base()) == 0_uz;
        else if constexpr (size_of_v<V> == 0) return std::true_type{};
        else return std::false_type{};
      }
      else
      {
        return stdex::ranges::begin(base()) == stdex::ranges::end(base());
      }
    }


    /**
     * \returns Returns the initial element.
     */
    constexpr decltype(auto)
    front() { return collections::get<0>(base()); }

    /// \overload
    constexpr decltype(auto)
    front() const { return collections::get<0>(base()); }

  private:

#ifdef __cpp_lib_ranges
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct back_gettable : std::false_type {};

    template<typename T>
#ifdef __cpp_lib_ranges
      requires values::fixed_value_compares_with<size_of<T>, stdex::dynamic_extent, &stdex::is_neq>
    struct back_gettable<T>
#else
    struct back_gettable<T, std::enable_if_t<
      values::fixed_value_compares_with<size_of<T>, stdex::dynamic_extent, &stdex::is_neq>>>
#endif
      : std::bool_constant<gettable<size_of_v<T> - 1_uz, T>> {};

  public:

    /**
     * \returns Returns the final element.
     */
#ifdef __cpp_concepts
    constexpr decltype(auto)
    back() requires (stdex::ranges::bidirectional_range<V> and stdex::ranges::common_range<V>) or back_gettable<V>::value
#else
    template<bool Enable = true, std::enable_if_t<Enable and
      (stdex::ranges::bidirectional_range<V> and stdex::ranges::common_range<V>) or back_gettable<V>::value, int> = 0>
    constexpr decltype(auto)
    back()
#endif
    {
      if constexpr (back_gettable<V>::value)
        return collections::get<size_of_v<V> - 1_uz>(base());
      else
        return stdex::ranges::view_interface<from_range<V>>::back();
    }

    /// \overload
#ifdef __cpp_concepts
    constexpr decltype(auto)
    back() const requires (stdex::ranges::bidirectional_range<V> and stdex::ranges::common_range<V>) or back_gettable<V>::value
#else
    template<bool Enable = true, std::enable_if_t<Enable and
      (stdex::ranges::bidirectional_range<V> and stdex::ranges::common_range<V>) or back_gettable<V>::value, int> = 0>
    constexpr decltype(auto)
    back() const
#endif
    {
      if constexpr (back_gettable<V>::value)
        return collections::get<size_of_v<V> - 1_uz>(base());
      else
        return stdex::ranges::view_interface<from_range<V>>::back();
    }

    
    /**
     * \brief Subscript operator
     * \note Performs no runtime bounds checking
     */
#ifdef __cpp_explicit_this_parameter
    template<typename Self, values::index I>
    constexpr decltype(auto)
    operator[](this Self&& self, I i)
    {
      static_assert(not values::size_compares_with<I, size_of<V>, &stdex::is_gteq>, "Index out of range");
      return collections::get_element(std::forward<Self>(self), std::move(i));
    }
#else
    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) &
    {
      static_assert(not values::size_compares_with<I, size_of<V>, &stdex::is_gteq>, "Index out of range");
      return collections::get_element(*this, std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const &
    {
      static_assert(not values::size_compares_with<I, size_of<V>, &stdex::is_gteq>, "Index out of range");
      return collections::get_element(*this, std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) && noexcept
    {
      static_assert(not values::size_compares_with<I, size_of<V>, &stdex::is_gteq>, "Index out of range");
      return collections::get_element(std::move(*this), std::move(i));
    }

    template<typename I, std::enable_if_t<values::index<I>, int> = 0>
    constexpr decltype(auto)
    operator[](I i) const && noexcept
    {
      static_assert(not values::size_compares_with<I, size_of<V>, &stdex::is_gteq>, "Index out of range");
      return collections::get_element(std::move(*this), std::move(i));
    }
#endif


    /**
     * \brief Get element i.
     * \note Performs no runtime bounds checking
     */
#ifdef __cpp_explicit_this_parameter
    template<std::size_t i>
    constexpr decltype(auto)
    get(this auto&& self) noexcept
    {
      using Ni = std::integral_constant<std::size_t, i>;
      static_assert(not values::size_compares_with<Ni, size_of<V>, &stdex::is_gteq>, "Index out of range");
      return detail_from_range::get_from_base<i>(std::forward<decltype(self)>(self).base());
    }
#else
    template<std::size_t i>
    constexpr decltype(auto)
    get() &
    {
      using Ni = std::integral_constant<std::size_t, i>;
      static_assert(not values::size_compares_with<Ni, size_of<V>, &stdex::is_gteq>, "Index out of range");
      return detail_from_range::get_from_base<i>(base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const &
    {
      using Ni = std::integral_constant<std::size_t, i>;
      static_assert(not values::size_compares_with<Ni, size_of<V>, &stdex::is_gteq>, "Index out of range");
      return detail_from_range::get_from_base<i>(base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() && noexcept
    {
      using Ni = std::integral_constant<std::size_t, i>;
      static_assert(not values::size_compares_with<Ni, size_of<V>, &stdex::is_gteq>, "Index out of range");
      return detail_from_range::get_from_base<i>(std::move(*this).base());
    }

    template<std::size_t i>
    constexpr decltype(auto)
    get() const && noexcept
    {
      using Ni = std::integral_constant<std::size_t, i>;
      static_assert(not values::size_compares_with<Ni, size_of<V>, &stdex::is_gteq>, "Index out of range");
      return detail_from_range::get_from_base<i>(std::move(*this).base());
    }
#endif

  private:

    RangeBox r_;

  };


  /// Deduction guide
  template<typename V>
  from_range(V&&) -> from_range<V>;


}


#ifdef __cpp_lib_ranges
namespace std::ranges
#else
namespace OpenKalman::stdex::ranges
#endif
{
  template<typename V>
  constexpr bool enable_borrowed_range<OpenKalman::collections::from_range<V>> =
    std::is_lvalue_reference_v<V> or enable_borrowed_range<OpenKalman::stdex::remove_cvref_t<V>>;
}


namespace std
{
  template<typename V>
  struct tuple_size<OpenKalman::collections::from_range<V>>
    : std::conditional_t<
      OpenKalman::values::fixed_value_compares_with<
        OpenKalman::collections::size_of<V>, OpenKalman::stdex::dynamic_extent, &OpenKalman::stdex::is_neq>,
      OpenKalman::collections::size_of<V>,
      monostate> {};


  template<std::size_t i, typename V>
  struct tuple_element<i, OpenKalman::collections::from_range<V>>
    : OpenKalman::collections::detail_from_range::get_elem<i, OpenKalman::stdex::remove_cvref_t<V>> {};
}


#endif
