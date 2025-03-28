/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2018-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of the DynamicDescriptor class.
 */

#ifndef OPENKALMAN_DYNAMICDESCRIPTOR_HPP
#define OPENKALMAN_DYNAMICDESCRIPTOR_HPP

#include <vector>
#include <tuple>
#include <functional>
#include "values/concepts/number.hpp"
#include "linear-algebra/coordinates/interfaces/coordinate_descriptor_traits.hpp"
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/traits/size_of.hpp"
#include "linear-algebra/coordinates/traits/euclidean_size_of.hpp"
#include "linear-algebra/coordinates/functions/get_size.hpp"
#include "linear-algebra/coordinates/functions/get_euclidean_size.hpp"
#include "linear-algebra/coordinates/functions/get_component_count.hpp"
#include "Dimensions.hpp"
#include "Any.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief A dynamic list of \ref coordinate::descriptor objects that can be defined or extended at runtime.
   * \details At compile time, the structure is treated if it has zero dimension.
   * \tparam Scalar The scalar type for elements associated with this object.
   */
#ifdef __cpp_concepts
  template<value::number Scalar>
#else
  template<typename Scalar>
#endif
  struct DynamicDescriptor
  {

  private:

#ifndef __cpp_concepts
    static_assert(value::number<Scalar>);
#endif

    template<std::size_t local_t = 0, typename Tup>
    constexpr void
    extend_table_fixed_t(const Tup& tup, std::size_t i, std::size_t e)
    {
      if constexpr (local_t < std::tuple_size_v<Tup>)
      {
        component_collection.emplace_back(std::get<local_t>(tup));
        using C = std::tuple_element_t<local_t, Tup>;
        component_start_indices.push_back(std::array{i, e});
        extend_table_fixed_t<local_t + 1>(tup, i + size_of_v<C>, e + euclidean_size_of_v<C>);
      }
    }


    template<std::size_t local_t = 0, std::size_t local_i = 0, typename Tup>
    constexpr void
    extend_table_fixed_i(std::size_t i, std::size_t e_start, std::size_t t, const Tup& tup)
    {
      if constexpr (local_t < std::tuple_size_v<Tup>)
      {
        using C = std::tuple_element_t<local_t, Tup>;
        if constexpr (local_i < size_of_v<C>)
        {
          index_table.emplace_back(t);
          extend_table_fixed_i<local_t, local_i + 1>(i + 1, e_start, t, tup);
        }
        else
        {
          extend_table_fixed_i<local_t + 1, 0>(i, e_start + euclidean_size_of_v<C>, t + 1, tup);
        }
      }
    }


    template<std::size_t local_t = 0, std::size_t local_e = 0, typename Tup>
    constexpr void
    extend_table_fixed_e(std::size_t i_start, std::size_t i_e, std::size_t t, const Tup& tup)
    {
      if constexpr (local_t < std::tuple_size_v<Tup>)
      {
        using C = std::tuple_element_t<local_t, Tup>;
        if constexpr (local_e < euclidean_size_of_v<C>)
        {
          euclidean_index_table.emplace_back(t);
          extend_table_fixed_e<local_t, local_e + 1>(i_start, i_e + 1, t, tup);
        }
        else
        {
          extend_table_fixed_e<local_t + 1, 0>(i_start + size_of_v<C>, i_e, t + 1, tup);
        }
      }
    }


    template<typename C>
    constexpr void
    fill_tables(C&& c, std::size_t i = 0, std::size_t e = 0, std::size_t t = 0)
    {
      if constexpr (fixed_pattern<C>)
      {
        auto coll = internal::get_component_collection(c);
        extend_table_fixed_t(coll, i, e);
        extend_table_fixed_i(i, e, t, coll);
        extend_table_fixed_e(i, e, t, coll);
      }
      else if constexpr (euclidean_pattern<C>)
      {
        std::size_t dim = get_size(c);
        component_collection.emplace_back(Dimensions{dim});
        component_start_indices.push_back(std::array {i, e});
        for (std::size_t N = i + dim; i < N; ++i, ++e)
        {
          index_table.emplace_back(t);
          euclidean_index_table.emplace_back(t);
        }
      }
      else // C is dynamic
      {
        for (const auto& j : c.component_collection) component_collection.emplace_back(j);
        for (const auto& [ji, je] : c.component_start_indices) component_start_indices.push_back(std::array {i + ji, e + je});
        for (const auto& j : c.index_table) index_table.emplace_back(t + j);
        for (const auto& j : c.euclidean_index_table) euclidean_index_table.emplace_back(t + j);
      }
    }


    constexpr void
    init(std::size_t i, std::size_t e, std::size_t t) {}


    template<typename C, typename...Cs>
    constexpr void
    init(std::size_t i, std::size_t e, std::size_t t, C&& c, Cs&&...cs)
    {
      fill_tables(std::forward<C>(c), i, e, t);
      init(i + get_size(c), e + get_euclidean_size(c), t + get_component_count(c), std::forward<Cs>(cs)...);
    }

  public:

    /**
     * \brief Constructor taking any number of \ref coordinate::pattern Cs.
     * \details \ref coordinate::pattern objects Cs can be either fixed or dynamic. If dynamic, the pattern must be either
     * untyped or the same type as this DynamicDescriptor.
     * \tparam Cs A list of sets of \ref coordinate::pattern for a set of indices.
     */
#ifdef __cpp_concepts
    template<pattern...Cs> requires (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::is_base_of_v<DynamicDescriptor, Cs> and ...))
#else
    template<typename...Cs, std::enable_if_t<(pattern<Cs> and ...) and (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::is_base_of_v<DynamicDescriptor, Cs> and ...)),
      int> = 0>
#endif
    explicit constexpr DynamicDescriptor(Cs&&...cs)
    {
      component_collection.reserve((0 + ... + get_component_count(cs)));
      component_start_indices.reserve((0 + ... + get_component_count(cs)));
      index_table.reserve((0 + ... + get_size(cs)));
      euclidean_index_table.reserve((0 + ... + get_euclidean_size(cs)));
      init(0, 0, 0, std::forward<Cs>(cs)...);
    }


    /**
     * \brief Default constructor.
     */
    constexpr DynamicDescriptor() = default;


    /**
     * \brief Extend the DynamicDescriptor by appending to the end of the list.
     */
#ifdef __cpp_concepts
    template<pattern B>
#else
    template<typename B>
#endif
    constexpr decltype(auto)
    operator+=(B&& b)
    {
      if (auto N = get_component_count(b); N > 1)
      {
        if (component_collection.capacity() < std::size(component_collection) + N)
          component_collection.reserve(component_collection.capacity() * 2);

        if (component_start_indices.capacity() < std::size(component_start_indices) + N)
          component_start_indices.reserve(component_start_indices.capacity() * 2);

        if (index_table.capacity() < (std::size(index_table) + get_size(b)))
          index_table.reserve(index_table.capacity() * 2);

        if (euclidean_index_table.capacity() < (euclidean_index_table.size() + get_euclidean_size(b)))
          euclidean_index_table.reserve(euclidean_index_table.capacity() * 2);
      }
      fill_tables(std::forward<B>(b), std::size(index_table), std::size(euclidean_index_table), std::size(component_collection));
      return *this;
    }


    /**
     * \brief Assign from another \ref coordinate::pattern.
     */
#ifdef __cpp_concepts
    template<pattern D> requires (not std::is_base_of_v<DynamicDescriptor, D>)
#else
    template<typename D, std::enable_if_t<pattern<D> and (not std::is_base_of_v<DynamicDescriptor, D>), int> = 0>
#endif
    constexpr DynamicDescriptor&
    operator=(const D& d)
    {
      component_collection.clear();
      component_start_indices.clear();
      index_table.clear();
      euclidean_index_table.clear();
      component_collection.reserve(get_component_count(d));
      component_start_indices.reserve(get_component_count(d));
      index_table.reserve(get_size(d));
      euclidean_index_table.reserve(get_euclidean_size(d));
      fill_tables(d);
      return *this;
    }


    /**
     * \brief Split the object into head and tail parts.
     */
#ifdef __cpp_concepts
  template<value::index Offset, value::index Extent> requires
    (value::dynamic<Offset> or Offset::value >= 0) and (value::dynamic<Extent> or Extent::value >= 0)
#else
  template<typename Offset, typename Extent>
#endif
    constexpr auto
    slice(const Offset& offset, const Extent& extent) const
    {
      DynamicDescriptor ret;

      auto i_first = index_table.begin() + offset;
      auto i_last = i_first + extent;

      if (i_first != index_table.end())
      {
        if (std::get<1>(*i_first) != 0)
          throw std::invalid_argument {"DynamicDescriptor slice: offset does not coincide with a sub-descriptor boundary"};

        if (extent > 0)
        {
          if (i_last != index_table.end() and std::get<1>(*i_last) != 0)
            throw std::invalid_argument {"DynamicDescriptor slice: offset + extent does not coincide with a sub-descriptor boundary"};

          std::size_t t_first_val = std::get<0>(*i_first);
          auto t_first = component_collection.begin() + t_first_val;
          auto t_last = i_last == index_table.end() ? component_collection.end() : component_collection.begin() + std::get<0>(*i_last);

          std::ptrdiff_t e_first_val = std::get<2>(*i_first);
          auto e_first = euclidean_index_table.begin() + e_first_val;
          auto e_last = i_last == index_table.end() ? euclidean_index_table.end() : euclidean_index_table.begin() + std::get<2>(*i_last);

          ret.component_collection.assign(t_first, t_last);

          ret.index_table.reserve(extent);
          for (auto i = i_first; i != i_last; i++)
          {
            ret.index_table.emplace_back(*i - t_first_val);
          }

          ret.euclidean_index_table.reserve(std::distance(e_first, e_last));
          for (auto e = e_first; e != e_last; e++)
          {
            ret.euclidean_index_table.emplace_back(*e - t_first_val);
          }
        }
      }

      return ret;
    }


    // ---------- friends ---------- //

#ifdef __cpp_concepts
    template<value::number>
#else
    template<typename>
#endif
    friend struct DynamicDescriptor;


/*#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename>
#endif
    friend struct interface::coordinate_set_traits;*/

  protected:

    // ---------- tables ---------- //

    std::vector<Any<Scalar>> component_collection;
    std::vector<std::array<std::size_t, 2>> component_start_indices;
    std::vector<std::size_t> index_table;
    std::vector<std::size_t> euclidean_index_table;

  };


  // -------------------------------------- //
  //            deduction guides            //
  // -------------------------------------- //

  /**
   * \brief Deduce scalar type when the constructor's first argument is a DynamicDescriptor.
   */
  template<typename S, typename Arg, typename...Args>
  DynamicDescriptor(DynamicDescriptor<S>&&, Arg&&, Args&&...) -> DynamicDescriptor<S>;

  /// \overload
  template<typename S, typename Arg, typename...Args>
  DynamicDescriptor(const DynamicDescriptor<S>&, Arg&&, Args&&...) -> DynamicDescriptor<S>;


} // namespace OpenKalman::coordinate


/*namespace OpenKalman::interface
{
  template<typename Scalar>
  struct coordinate_set_traits<coordinate::DynamicDescriptor<Scalar>>
  {
  private:

    using T = coordinate::DynamicDescriptor<Scalar>;
    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

  public:

    static constexpr bool is_specialized = true;


    using scalar_type = Scalar;


    template<typename Arg>
    static constexpr decltype(auto)
    component_collection(Arg&& arg) { return std::forward<Arg>(arg).component_collection; }


#ifdef __cpp_concepts
    template<typename Arg, value::index N>
#else
    template<typename Arg, typename N, std::enable_if_t<value::index<N>, int> = 0>
#endif
    static constexpr auto
    component_start_indices(Arg&& arg, N n)
    {
      return std::forward<Arg>(arg).component_start_indices[static_cast<std::size_t>(n)];
    }


    template<typename Arg>
    static constexpr decltype(auto)
    index_table(Arg&& arg)
    {
      return std::forward<Arg>(arg).index_table;
    }


    template<typename Arg>
    static constexpr decltype(auto)
    euclidean_index_table(Arg&& arg)
    {
      return std::forward<Arg>(arg).euclidean_index_table;
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T& t, const auto& g, const value::index auto& euclidean_local_index)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<scalar_type>; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, scalar_type> and value::index<L>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index)
#endif
    {
      auto local_e = static_cast<std::size_t>(euclidean_local_index);
      auto c = t.euclidean_index_table[local_e];
      auto [comp_i, comp_e] = component_start_indices(t, c);
      auto new_g = [&g, comp_i](std::size_t i) { return g(comp_i + i); };
      return coordinate::to_euclidean_element(t.component_collection[c], new_g, local_e - comp_e);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T& t, const auto& g, const value::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<scalar_type>; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, scalar_type>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      auto local_i = static_cast<std::size_t>(local_index);
      auto c = t.index_table[local_i];
      auto [comp_i, comp_e] = component_start_indices(t, c);
      auto new_g = [&g, comp_e](std::size_t e) { return g(comp_e + e); };
      return coordinate::from_euclidean_element(t.component_collection[c], new_g, local_i - comp_i);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T& t, const auto& g, const value::index auto& local_index)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<scalar_type>; }
#else
    template<typename Getter, typename L, std::enable_if_t<value::index<L> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, scalar_type>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index)
#endif
    {
      auto local_i = static_cast<std::size_t>(local_index);
      auto c = t.index_table[local_i];
      auto [comp_i, comp_e] = component_start_indices(t, c);
      auto new_g = [&g, comp_i](std::size_t i) { return g(comp_i + i); };
      return coordinate::get_wrapped_component(t.component_collection[c], new_g, local_i - comp_i);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const scalar_type& x, const value::index auto& local_index)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename L, std::enable_if_t<value::index<L> and
      std::is_invocable<const Setter&, const scalar_type&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const scalar_type& x, const L& local_index)
#endif
    {
      auto local_i = static_cast<std::size_t>(local_index);
      auto c = t.index_table[local_i];
      auto [comp_i, comp_e] = component_start_indices(t, c);
      auto new_g = [&g, comp_i](std::size_t i) { return g(comp_i + i); };
      auto new_s = [&s, comp_i](const scalar_type& x, std::size_t i) { s(x, comp_i + i); };
      coordinate::set_wrapped_component(t.component_collection[c], new_s, new_g, x, local_i - comp_i);
    }

  };

} // namespace OpenKalman::interface */


#endif //OPENKALMAN_DYNAMICDESCRIPTOR_HPP
