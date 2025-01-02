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
#include <complex>
#include <functional>
#include <typeindex>

#include "linear-algebra/values/values.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_euclidean_dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/functions/get_vector_space_descriptor_component_count_of.hpp"
#include "StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/interfaces/index.hpp"
#include "internal/AnyAtomicVectorSpaceDescriptor.hpp"

namespace OpenKalman::descriptor
{
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

  public:

    /**
     * \brief Default constructor.
     */
    constexpr DynamicDescriptor() = default;


    /**
     * \brief Constructor taking any number of \ref vector_space_descriptor Cs.
     * \details \ref vector_space_descriptor objects Cs can be either fixed or dynamic. If dynamic, the vector_space_descriptor must be either
     * untyped or the same type as this DynamicDescriptor.
     * \tparam Cs A list of sets of \ref vector_space_descriptor for a set of indices.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor...Cs> requires (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::is_base_of_v<DynamicDescriptor, Cs> and ...)) and
      (sizeof...(Cs) == 1 or ((static_vector_space_descriptor<Cs> or dynamic_vector_space_descriptor<Cs>) and ...))
#else
    template<typename...Cs, std::enable_if_t<(vector_space_descriptor<Cs> and ...) and (sizeof...(Cs) > 0) and
      (sizeof...(Cs) != 1 or (not std::is_base_of_v<DynamicDescriptor, Cs> and ...)) and
      (sizeof...(Cs) == 1 or ((static_vector_space_descriptor<Cs> or dynamic_vector_space_descriptor<Cs>) and ...)),
      int> = 0>
#endif
    explicit constexpr DynamicDescriptor(Cs&&...cs)
    {
      dynamic_types.reserve((0 + ... + get_vector_space_descriptor_component_count_of(cs)));
      index_table.reserve((0 + ... + get_dimension_size_of(cs)));
      euclidean_index_table.reserve((0 + ... + get_euclidean_dimension_size_of(cs)));
      fill_tables(0, 0, 0, std::forward<Cs>(cs)...);
    }


    /**
     * \brief Constructor taking another DynamicDescriptor and other list of \ref vector_space_descriptor objects.
     * \details This forwards the first argument to the move or forward constructor and then extends by adding the other arguments.
     * \param c A DynamicDescriptor object
     * \param cn A first \ref vector_space_descriptor.
     * \param cs A list of \ref vector_space_descriptor objects.
     */
#ifdef __cpp_concepts
    template<typename C, vector_space_descriptor C0, vector_space_descriptor...Cn> requires
      std::derived_from<std::decay_t<C>, DynamicDescriptor>
#else
    template<typename C, typename C0, typename...Cn, std::enable_if_t<
      std::is_base_of_v<DynamicDescriptor, std::decay_t<C>> and (vector_space_descriptor<Cn> and ...), int> = 0>
#endif
    constexpr DynamicDescriptor(C&& c, C0&& c0, Cn&&...cn) : DynamicDescriptor(std::forward<C>(c))
    {
      extend(std::forward<C0>(c0), std::forward<Cn>(cn)...);
    }


    /**
     * \brief Assign from another \ref vector_space_descriptor.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor D> requires (not std::is_base_of_v<DynamicDescriptor, D>)
#else
    template<typename D, std::enable_if_t<vector_space_descriptor<D> and (not std::is_base_of_v<DynamicDescriptor, D>), int> = 0>
#endif
    constexpr DynamicDescriptor&
    operator=(const D& d)
    {
      dynamic_types.clear();
      index_table.clear();
      euclidean_index_table.clear();
      dynamic_types.reserve(get_vector_space_descriptor_component_count_of(d));
      index_table.reserve(get_dimension_size_of(d));
      euclidean_index_table.reserve(get_euclidean_dimension_size_of(d));
      fill_tables(0, 0, 0, d);
      return *this;
    }


    /**
     * \brief Extend the DynamicDescriptor by appending to the end of the list.
     * \tparam Cs One or more \ref vector_space_descriptor objects
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor...Cs> requires (sizeof...(Cs) > 0) and
      ((static_vector_space_descriptor<Cs> or euclidean_vector_space_descriptor<Cs> or std::same_as<std::decay_t<Cs>, DynamicDescriptor>) and ...)
#else
    template<typename...Cs, std::enable_if_t<(vector_space_descriptor<Cs> and ...) and (sizeof...(Cs) > 0) and
      ((static_vector_space_descriptor<Cs> or euclidean_vector_space_descriptor<Cs> or std::is_same_v<std::decay_t<Cs>, DynamicDescriptor>) and ...), int> = 0>
#endif
    constexpr DynamicDescriptor&
    extend(Cs&&...cs)
    {
      if (auto N = (0 + ... + get_vector_space_descriptor_component_count_of(cs)); N > 1)
      {
        if (dynamic_types.capacity() < std::size(dynamic_types) + N)
          dynamic_types.reserve(dynamic_types.capacity() * 2);

        if (index_table.capacity() < (std::size(index_table) + ... + get_dimension_size_of(cs)))
          index_table.reserve(index_table.capacity() * 2);

        if (euclidean_index_table.capacity() < (euclidean_index_table.size() + ... + get_euclidean_dimension_size_of(cs)))
          euclidean_index_table.reserve(euclidean_index_table.capacity() * 2);
      }
      fill_tables(std::size(index_table), std::size(euclidean_index_table), std::size(dynamic_types), std::forward<Cs>(cs)...);
      return *this;
    }

    // ---------- iterators ---------- //

    /**
     * \return Iterator marking the beginning of a vector containing a set of DynamicTypedVectorSpaceDescriptor objects.
     */
    constexpr auto
    begin() const { return std::begin(dynamic_types); }


    /**
     * \return Iterator marking the end of a vector containing a set of DynamicTypedVectorSpaceDescriptor objects.
     */
    constexpr auto
    end() const { return std::end(dynamic_types); }

  private:

    // ---------- extending tables ---------- //

    /*
     * \internal
     * \tparam euclidean Whether the relevant vector is in Euclidean space (true) or not (false)
     * \tparam local_index The local index for indices associated with each of dynamic_types (resets to 0 when t increments)
     * \tparam C The current index type being processed
     * \tparam Cs Remaining index types to process
     * \param i The row index
     * \param t The component index within dynamic_types
     * \param start The start location in the corresponding Euclidean or non-Euclidean vector
     * \return A tuple of tuples of {t, local_index, start}
     */
    template<bool euclidean, std::size_t local_index, typename C, typename...Cs>
    constexpr void
    extend_table_fixed(std::size_t i, std::size_t t, std::size_t start)
    {
      constexpr auto i_size = dimension_size_of_v<C>;
      constexpr auto e_size = euclidean_dimension_size_of_v<C>;
      if constexpr (local_index >= (euclidean ? e_size : i_size))
      {
        extend_table_fixed<euclidean, 0, Cs...>(i, t + 1, start + (euclidean ? i_size : e_size));
      }
      else
      {
        (euclidean ? euclidean_index_table : index_table).emplace_back(t, local_index, start);
        extend_table_fixed<euclidean, local_index + 1, C, Cs...>(i + 1, t, start);
      }
    }


    // \overload
    template<bool euclidean, std::size_t local_index>
    constexpr void
    extend_table_fixed(std::size_t i, std::size_t t, std::size_t start) {}


    template<typename C>
    constexpr void
    fill_tables_fixed(std::size_t i, std::size_t i_e, std::size_t t, const C& c)
    {
      dynamic_types.emplace_back(internal::AnyAtomicVectorSpaceDescriptor<Scalar> {c});
      extend_table_fixed<false, 0, C>(i, t, i_e);
      extend_table_fixed<true, 0, C>(i_e, t, i);
    }


    template<typename...Cs>
    constexpr void
    fill_tables_fixed(std::size_t i, std::size_t i_e, std::size_t t, const StaticDescriptor<Cs...>&)
    {
      (dynamic_types.emplace_back(internal::AnyAtomicVectorSpaceDescriptor<Scalar> {Cs{}}), ...);
      extend_table_fixed<false, 0, Cs...>(i, t, i_e);
      extend_table_fixed<true, 0, Cs...>(i_e, t, i);
    }


    template<typename C, typename...Cs>
    constexpr void
    fill_tables(std::size_t i, std::size_t i_e, std::size_t t, C&& c, Cs&&...cs)
    {
      if constexpr (static_vector_space_descriptor<C>)
      {
        fill_tables_fixed(i, i_e, t, internal::canonical_equivalent(c));
        fill_tables(
          i + get_dimension_size_of(c),
          i_e + get_euclidean_dimension_size_of(c),
          t + get_vector_space_descriptor_component_count_of(c),
          std::forward<Cs>(cs)...);
      }
      else // if constexpr (dynamic_vector_space_descriptor<C>)
      {
        if constexpr (euclidean_vector_space_descriptor<C>)
        {
          auto N = t + get_dimension_size_of(c);
          for (; t < N; ++i, ++i_e, ++t)
          {
            dynamic_types.emplace_back(internal::AnyAtomicVectorSpaceDescriptor<Scalar> {Dimensions<1>{}});
            index_table.emplace_back(t, 0, i_e);
            euclidean_index_table.emplace_back(t, 0, i);
          }
          fill_tables(i, i_e, t, std::forward<Cs>(cs)...);
        }
        else // C is DynamicDescriptor.
        {
          auto new_i = i + c.index_table.size();
          auto new_i_e = i_e + c.euclidean_index_table.size();
          auto new_t = t + c.dynamic_types.size();

          for (auto&& j : c.dynamic_types)
            dynamic_types.emplace_back(std::forward<decltype(j)>(j));

          for (auto&& j : c.index_table)
            index_table.emplace_back(
              t + std::get<0>(std::forward<decltype(j)>(j)),
              std::get<1>(std::forward<decltype(j)>(j)),
              i_e + std::get<2>(std::forward<decltype(j)>(j)));

          for (auto&& j : c.euclidean_index_table)
            euclidean_index_table.emplace_back(
              t + std::get<0>(std::forward<decltype(j)>(j)),
              std::get<1>(std::forward<decltype(j)>(j)),
              i + std::get<2>(std::forward<decltype(j)>(j)));

          fill_tables(new_i, new_i_e, new_t, std::forward<Cs>(cs)...);
        }
      }
    }


    constexpr void
    fill_tables(std::size_t, std::size_t, std::size_t) {}


    // ---------- comparison ---------- //

    template<std::size_t n, std::size_t ni, std::size_t ne, typename It, typename EndIt>
    constexpr auto subtract(It, EndIt, StaticDescriptor<>) const
    {
      DynamicDescriptor ret;
      ret.dynamic_types.assign(dynamic_types.begin(), dynamic_types.end() - n);
      ret.index_table.assign(index_table.begin(), index_table.end() - ni);
      ret.euclidean_index_table.assign(euclidean_index_table.begin(), euclidean_index_table.end() - ne);
      return ret;
    }


    template<std::size_t n, std::size_t ni, std::size_t ne, typename It, typename EndIt, typename C, typename...Cs>
    constexpr auto subtract(It it, EndIt endit, StaticDescriptor<C, Cs...>) const
    {
      if (it == endit or not descriptor::internal::are_equivalent(*--endit, C{}))
        throw std::invalid_argument {"Subtraction of incompatible vector_space_descriptor values"};
      else
        return subtract<n + 1, ni + dimension_size_of_v<C>, ne + euclidean_dimension_size_of_v<C>>(it, endit, StaticDescriptor<Cs...>{});
    }

  public:

    /**
     * \brief Minus operator when the operand is a \ref static_vector_space_descriptor
     */
/*#ifdef __cpp_concepts
    template<static_vector_space_descriptor B>
#else
    template<typename B, std::enable_if_t<static_vector_space_descriptor<B>, int> = 0>
#endif
    constexpr auto operator-(const B& b) const
    {
      if constexpr (dimension_size_of_v<B> == 0) return *this;
      else
      {
        using F = static_reverse_t<std::decay_t<decltype(internal::canonical_equivalent(b))>>;
        return subtract<0,0,0>(dynamic_types.begin(), dynamic_types.end(), F{});
      }
    }*/


    /**
     * \overload
     * \brief Minus operator when the operand is a dynamic \ref euclidean_vector_space_descriptor
     */
#ifdef __cpp_concepts
    template<euclidean_vector_space_descriptor B> requires dynamic_vector_space_descriptor<B>
#else
    template<typename B, std::enable_if_t<euclidean_vector_space_descriptor<B> and dynamic_vector_space_descriptor<B>, int> = 0>
#endif
    constexpr auto operator-(const B& b) const
    {
      // Compare the tails of a and b, from right to left
      auto i = dynamic_types.end();
      std::size_t j = 0;
      for (; j < get_dimension_size_of(b); ++j)
      {
        if (i == dynamic_types.begin() or not descriptor::get_vector_space_descriptor_is_euclidean(*--i))
          throw std::invalid_argument {"Subtraction of incompatible dynamic_vector_space_descriptor values"};
      }
      // Construct a dynamic vector space descriptor from the remainder of a
      DynamicDescriptor ret;
      ret.dynamic_types.assign(dynamic_types.begin(), i);
      ret.index_table.assign(index_table.begin(), index_table.end() - j);
      ret.euclidean_index_table.assign(euclidean_index_table.begin(), euclidean_index_table.end() - j);
      return ret;
    }


    /**
     * \overload
     * \brief Minus operator when the operand is another \ref DynamicDescriptor
     */
    template<typename S>
    friend constexpr auto operator-(const DynamicDescriptor& a, const DynamicDescriptor<S>& b)
    {
      // Compare the tails of a and b, from right to left
      auto i = a.dynamic_types.end();
      auto ii = a.index_table.end();
      auto ie = a.euclidean_index_table.end();
      for (auto j = b.dynamic_types.end(); j != b.dynamic_types.begin(); )
      {
        if (i == a.begin() or not internal::are_equivalent(*--i, *--j))
          throw std::invalid_argument {"Subtraction of incompatible dynamic_vector_space_descriptor values"};
        ii -= get_dimension_size_of(*j);
        ie -= get_euclidean_dimension_size_of(*j);
      }
      // Construct a dynamic vector space descriptor from the remainder of a
      DynamicDescriptor ret;
      ret.dynamic_types.assign(a.dynamic_types.begin(), i);
      ret.index_table.assign(a.index_table.begin(), ii);
      ret.euclidean_index_table.assign(a.euclidean_index_table.begin(), ie);
      return ret;
    }


    // ---------- operator+= is the same as extend ---------- //

#ifdef __cpp_concepts
    template<vector_space_descriptor B>
#else
    template<typename B>
#endif
    constexpr decltype(auto) operator+=(B&& b) { return extend(std::forward<B>(b)); }


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
      //std::cout << "size: " << std::size(index_table) << "; offset: " << offset << "; extent: " << extent << std::endl;
      //for (const auto& [i, local, e_start] : index_table)
      //  std::cout << "(" << i << ", " << local << ", " << e_start << ")" << std::endl;

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
          auto t_first = dynamic_types.begin() + t_first_val;
          auto t_last = i_last == index_table.end() ? dynamic_types.end() : dynamic_types.begin() + std::get<0>(*i_last);

          std::ptrdiff_t e_first_val = std::get<2>(*i_first);
          auto e_first = euclidean_index_table.begin() + e_first_val;
          auto e_last = i_last == index_table.end() ? euclidean_index_table.end() : euclidean_index_table.begin() + std::get<2>(*i_last);

          ret.dynamic_types.assign(t_first, t_last);

          ret.index_table.reserve(extent);
          for (auto i = i_first; i != i_last; i++)
          {
            ret.index_table.emplace_back(
              std::get<0>(*i) - t_first_val,
              std::get<1>(*i),
              std::get<2>(*i) - e_first_val);
          }

          ret.euclidean_index_table.reserve(std::distance(e_first, e_last));
          for (auto e = e_first; e != e_last; e++)
          {
            ret.euclidean_index_table.emplace_back(
              std::get<0>(*e) - t_first_val,
              std::get<1>(*e),
              std::get<2>(*e) - offset);
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


    template<typename T>
    friend struct interface::vector_space_traits;

  protected:

    // ---------- tables ---------- //

    std::vector<internal::AnyAtomicVectorSpaceDescriptor<Scalar>> dynamic_types;
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> index_table;
    std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> euclidean_index_table;

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


} // namespace OpenKalman::descriptor


namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief traits for DynamicDescriptor.
   */
  template<typename Scalar>
  struct vector_space_traits<descriptor::DynamicDescriptor<Scalar>>
  {
  private:

    using T = descriptor::DynamicDescriptor<Scalar>;
    using Getter = std::function<Scalar(std::size_t)>;
    using Setter = std::function<void(const Scalar&, std::size_t)>;

  public:

    static constexpr auto
    size(const T& t) { return t.index_table.size(); }


    static constexpr auto
    euclidean_size(const T& t) { return t.euclidean_index_table.size(); }


    static constexpr const auto&
    collection(const T& t) { return t.dynamic_types; }


    static constexpr auto
    is_euclidean(const T& t)
    {
      for (const auto& i : t.dynamic_types)
        if (not descriptor::get_vector_space_descriptor_is_euclidean(i)) return false;
      return true;
    }


    template<typename Arg>
    static constexpr auto
    subtract(const T& t, const Arg& arg)
    {
      return t.operator-(arg);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    to_euclidean_component(const T& t, const auto& g, const value::index auto& euclidean_local_index, const value::index auto& start)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<Scalar>; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, Scalar> and value::index<L> and value::index<S>, int> = 0>
    static constexpr auto
    to_euclidean_component(const T& t, const Getter& g, const L& euclidean_local_index, const S& start)
#endif
    {
      auto [tp, comp_euclidean_local_index, comp_start] = t.euclidean_index_table[euclidean_local_index];
      return descriptor::to_euclidean_element(t.dynamic_types[tp], g, comp_euclidean_local_index, start + comp_start);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    from_euclidean_component(const T& t, const auto& g, const value::index auto& local_index, const value::index auto& euclidean_start)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<Scalar>; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, Scalar>, int> = 0>
    static constexpr auto
    from_euclidean_component(const T& t, const Getter& g, const L& local_index, const S& euclidean_start)
#endif
    {
      auto [tp, comp_local_index, comp_euclidean_start] = t.index_table[local_index];
      return descriptor::from_euclidean_element(t.dynamic_types[tp], g, comp_local_index, euclidean_start + comp_euclidean_start);
    }


#ifdef __cpp_concepts
    static constexpr value::value auto
    get_wrapped_component(const T& t, const auto& g, const value::index auto& local_index, const value::index auto& start)
    requires requires(std::size_t i){ {g(i)} -> std::convertible_to<Scalar>; }
#else
    template<typename Getter, typename L, typename S, std::enable_if_t<value::index<L> and value::index<S> and
      std::is_convertible_v<typename std::invoke_result<const Getter&, std::size_t>::type, Scalar>, int> = 0>
    static constexpr auto
    get_wrapped_component(const T& t, const Getter& g, const L& local_index, const S& start)
#endif
    {
      auto [tp, comp_local_index, comp_start] = t.index_table[local_index];
      return descriptor::get_wrapped_component(t.dynamic_types[tp], g, comp_local_index, start + local_index - comp_local_index);
    }


#ifdef __cpp_concepts
    static constexpr void
    set_wrapped_component(const T& t, const auto& s, const auto& g, const Scalar& x,
      const value::index auto& local_index, const value::index auto& start)
    requires requires(std::size_t i){ s(x, i); s(g(i), i); }
#else
    template<typename Setter, typename Getter, typename L, typename S, std::enable_if_t<
      value::index<L> and value::index<S> and
      std::is_invocable<const Setter&, const Scalar&, std::size_t>::value and
      std::is_invocable<const Setter&, typename std::invoke_result<const Getter&, std::size_t>::type, std::size_t>::value, int> = 0>
    static constexpr void
    set_wrapped_component(const T& t, const Setter& s, const Getter& g, const Scalar& x, const L& local_index, const S& start)
#endif
    {
      auto [tp, comp_local_index, comp_start] = t.index_table[local_index];
      descriptor::set_wrapped_component(t.dynamic_types[tp], s, g, x, comp_local_index, start + local_index - comp_local_index);
    }

  };

} // namespace OpenKalman::interface


#endif //OPENKALMAN_DYNAMICDESCRIPTOR_HPP
