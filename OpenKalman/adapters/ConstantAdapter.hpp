/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for ConstantAdapter
 */

#ifndef OPENKALMAN_CONSTANTADAPTER_HPP
#define OPENKALMAN_CONSTANTADAPTER_HPP

#include <type_traits>
#include <algorithm>

namespace OpenKalman
{
  // ----------------- //
  //  ConstantAdapter  //
  // ----------------- //

#ifdef __cpp_concepts
  template<indexible PatternMatrix, scalar_constant Scalar, auto...constant>
    requires (sizeof...(constant) == 0) or requires { Scalar {constant...}; }
#else
  template<typename PatternMatrix, typename Scalar, auto...constant>
#endif
  struct ConstantAdapter : internal::library_base_t<ConstantAdapter<PatternMatrix, Scalar, constant...>, PatternMatrix>
  {

  private:

#ifndef __cpp_concepts
    static_assert(indexible<PatternMatrix>);
    static_assert(scalar_constant<Scalar>);
    static_assert(sizeof...(constant) == 0 or std::is_constructible_v<Scalar, decltype(constant)...>);
#endif

    using MyConstant = std::conditional_t<sizeof...(constant) == 0, Scalar, values::ScalarConstant<Scalar, constant...>>;
    using MyScalarType = std::decay_t<decltype(get_scalar_constant_value(std::declval<MyConstant>()))>;

    using AllDescriptorsType = decltype(all_vector_space_descriptors(std::declval<PatternMatrix>()));
    using DescriptorCollection = std::conditional_t<
      vector_space_descriptor_tuple<AllDescriptorsType>, 
      AllDescriptorsType, 
      std::vector<DynamicDescriptor<MyScalarType>>>;


    template<std::size_t N = 0>
    constexpr auto make_descriptors_tuple() 
    { 
      if constexpr (N >= std::tuple_size_v<DescriptorCollection>) 
        return std::tuple {}; 
      else 
        return std::tuple_cat(
          std::forward_as_tuple(std::tuple_element_t<N, DescriptorCollection>{}), 
          make_descriptors_tuple<N + 1>());
    }

    template<std::size_t N = 0, typename D, typename...Ds>
    constexpr auto make_descriptors_tuple(D&& d, Ds&&...ds)
    {
      if constexpr (N >= std::tuple_size_v<DescriptorCollection>) 
      {
        if constexpr (dynamic_vector_space_descriptor<D>) if (d != Axis{}) throw std::invalid_argument {
          "Too many elements in vector space descriptors_collection_tuple argument of a constant adapter"};
        return make_descriptors_tuple<N + 1>(std::forward<Ds>(ds)...); 
      }
      else if constexpr (fixed_vector_space_descriptor<std::tuple_element_t<N, DescriptorCollection>>)
      {
        using E = std::tuple_element_t<N, DescriptorCollection>;
        if constexpr (dynamic_vector_space_descriptor<D>) if (d != E{}) throw std::invalid_argument {
          "Invalid dynamic element in vector_space_descriptor_tuple argument of a constant adapter"};
        return std::tuple_cat(std::forward_as_tuple(E{}),
          make_descriptors_tuple<N + 1>(std::forward<Ds>(ds)...));
      }
      else // if constexpr (dynamic_vector_space_descriptor<std::tuple_element_t<N, DescriptorCollection>>)
      {
        return std::tuple_cat(std::forward_as_tuple(std::forward<D>(d)),
          make_descriptors_tuple<N + 1>(std::forward<Ds>(ds)...));
      }
    }
    
    
    template<std::size_t N = 0, typename Iterator, typename Sentinel>
    constexpr auto make_descriptors_tuple_from_range(Iterator it, const Sentinel& end)
    {
      if constexpr (N >= std::tuple_size_v<DescriptorCollection>)
      {
        for (; it != end; ++it) if (*it != Axis{}) throw std::invalid_argument {
          "Too many elements in vector space descriptors_collection range argument of a constant adapter"};
        return std::tuple {};
      }
      else if constexpr (fixed_vector_space_descriptor<std::tuple_element_t<N, DescriptorCollection>>)
      {
        using E = std::tuple_element_t<N, DescriptorCollection>;
        if (it == end)
        {
          if (not equivalent_to<E, Axis>) throw std::invalid_argument {
            "Too few elements in vector space descriptors_collection range argument of a constant adapter"};
          return std::tuple_cat(std::forward_as_tuple(E{}),
            make_descriptors_tuple_from_range<N + 1>(std::move(it), end));
        }
        else
        {
          if constexpr (dynamic_vector_space_descriptor<decltype(*it)>) 
            if (*it != E{}) throw std::invalid_argument {
              "Invalid dynamic element in vector_space_descriptor_collection range argument of a constant adapter"};
          return std::tuple_cat(std::forward_as_tuple(E{}),
            make_descriptors_tuple_from_range<N + 1>(++it, end));
        }
      }
      else // if constexpr (dynamic_vector_space_descriptor<std::tuple_element_t<N, DescriptorCollection>>)
      {
        using E = std::tuple_element_t<N, DescriptorCollection>;
        if (it == end) throw std::invalid_argument {
          "Too few elements in vector space descriptors_collection range argument of a constant adapter"};
        std::tuple_element_t<N, DescriptorCollection> e {*it};
        return std::tuple_cat(std::forward_as_tuple(std::move(e)),
          make_descriptors_tuple_from_range<N + 1>(++it, end));
      }
    }
    
    
    template<typename Descriptors>
    constexpr auto make_descriptors_collection(Descriptors&& descriptors)
    {
      if constexpr (vector_space_descriptor_tuple<Descriptors>)
      {
        return std::apply([](auto&& D, auto&&...Ds){ 
            return make_descriptors_tuple(std::forward<D>(d), std::forward<Ds>(ds)...); 
          }, std::forward<Descriptors>(descriptors));
      }
      else if constexpr (vector_space_descriptor_tuple<DescriptorCollection>)
      {
        return make_descriptors_tuple_from_range(descriptors.begin(), descriptors.end()); 
      }
      else
      {
        DescriptorCollection ret;
        std::copy(descriptors.begin(), descriptors.end(), ret.begin());
        return ret;
      }
    }

  public:

    /**
     * \brief Construct from \ref scalar_constant and a \ref vector_space_descriptor_collection
     */
#ifdef __cpp_lib_ranges
    template<scalar_constant C, vector_space_descriptor_collection Descriptors> requires 
      std::constructible_from<MyConstant, C&&> and 
      compatible_with_vector_space_descriptor_collection<PatternMatrix, Descriptors>
#else
    template<typename C, typename Descriptors, std::enable_if_t<
      scalar_constant<C> and vector_space_descriptor_collection<Descriptors> and 
      std::is_constructible_v<MyConstant, C&&> and 
      compatible_with_vector_space_descriptor_collection<PatternMatrix, Descriptors>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(C&& c, Descriptors&& descriptors) 
      : my_constant {std::forward<C>(c)}, 
        descriptor_collection {make_descriptors_collection(std::forward<Descriptors>(descriptors))} {}


    /**
     * \overload
     * \brief Same as above, where the constant is known at compile time.
     */
#ifdef __cpp_lib_ranges
    template<vector_space_descriptor_collection Descriptors> requires 
      static_constant<MyConstant, ConstantType::static_constant> and 
      compatible_with_vector_space_descriptor_collection<PatternMatrix, Descriptors>
#else
    template<typename Descriptors, std::enable_if_t<
      vector_space_descriptor_collection<Descriptors> and 
      static_constant<MyConstant, ConstantType::static_constant> and 
      compatible_with_vector_space_descriptor_collection<PatternMatrix, Descriptors>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(Descriptors&& descriptors) 
      : descriptor_collection {make_descriptors_collection(std::forward<Descriptors>(descriptors))} {}

  
    /**
     * \brief Construct based on a \ref scalar_constant and the shape of an \ref indexible reference object.
     * \details 
     * The following constructs a 2-by-3 ConstantAdapter with constant 5.
     * \code
     * ConstantAdapter {std::integral_constant<int, 5>{}, eigen_matrix_t<double, 2, 3>{})
     * ConstantAdapter {std::integral_constant<int, 5>{}, eigen_matrix_t<double, 2, Eigen::Dynamic>(2, 3))
     * ConstantAdapter {5, eigen_matrix_t<double, 2, 3>{})
     * ConstantAdapter {5, eigen_matrix_t<double, Eigen::Dynamic, 3>(2, 3))
     * \endcode
     */
#ifdef __cpp_concepts
    template<scalar_constant C, vector_space_descriptors_may_match_with<PatternMatrix> Arg> requires
      std::constructible_from<MyConstant, C&&>
#else
    template<typename C, typename Arg, std::enable_if_t<
      scalar_constant<C> and vector_space_descriptors_may_match_with<Arg, PatternMatrix> and 
      std::is_constructible_v<MyConstant, C&&>, int> = 0>
#endif
    constexpr ConstantAdapter(C&& c, const Arg& arg) :
      my_constant {std::forward<C>(c)}, 
      descriptor_collection {make_descriptors_collection(all_vector_space_descriptors(arg))} {}


    /**
     * \overload 
     * \brief Same as above, where the constant is known at compile time.
     */
#ifdef __cpp_concepts
    template<vector_space_descriptors_may_match_with<PatternMatrix> Arg> requires
      (not std::is_base_of_v<ConstantAdapter, Arg>) and 
      static_constant<MyConstant, ConstantType::static_constant> 
#else
    template<typename Arg, std::enable_if_t<
      vector_space_descriptors_may_match_with<Arg, PatternMatrix> and 
      (not std::is_base_of_v<ConstantAdapter, Arg>) and 
      static_constant<MyConstant, ConstantType::static_constant>, int> = 0>
#endif
    constexpr ConstantAdapter(const Arg& arg) :
      descriptor_collection {make_descriptors_collection(all_vector_space_descriptors(arg))} {}


    /**
     * \brief Construct from another constant object.
     */
#ifdef __cpp_concepts
    template<constant_matrix Arg> requires
      (not std::is_base_of_v<ConstantAdapter, Arg>) and 
      vector_space_descriptors_may_match_with<Arg, PatternMatrix> and
      std::constructible_from<MyConstant, constant_coefficient<Arg>>
#else
    template<typename Arg, std::enable_if_t<constant_matrix<Arg> and
      (not std::is_base_of_v<ConstantAdapter, Arg>) and 
      vector_space_descriptors_may_match_with<Arg, PatternMatrix> and
      std::is_constructible_v<MyConstant, constant_coefficient<Arg>>, int> = 0>
#endif
    constexpr ConstantAdapter(const Arg& arg) :
      my_constant {constant_coefficient {arg}}, 
      descriptor_collection {make_descriptors_collection(all_vector_space_descriptors(arg))} {}


    /**
     * \brief Construct using a full set of \ref vector_space_descriptor objects.
     * \details For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>, 5>(std::integral_constant<int, 5>{}, 2, 3)
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, 3>>(5., 2, 3)
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, Eigen::Dynamic>>(5., 2, 3)
     * \endcode
     */
#ifdef __cpp_concepts
    template<scalar_constant C, vector_space_descriptor...Ds> requires 
      std::constructible_from<MyConstant, C&&> and
      compatible_with_vector_space_descriptors<PatternMatrix, Ds...>
#else
    template<typename C, typename...Ds, std::enable_if_t<
      scalar_constant<C> and (vector_space_descriptor<Ds> and ...) and 
      std::is_constructible_v<MyConstant, C&&> and
      compatible_with_vector_space_descriptors<PatternMatrix, Ds...>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(C&& c, Ds&&...ds)
      : my_constant {std::forward<C>(c)}, 
        descriptor_collection {make_descriptors_collection(std::forward_as_tuple(std::forward<Ds>(ds)...))} {}


    /**
     * \overload
     * \brief Same as above, where the constant is known at compile time.
     * \tparam Ds A set of \ref vector_space_descriptor objects corresponding to PatternMatrix.
     * \details
     * For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, 1, 1>, 5>()
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>, 5>(2, std::integral_constant<int, 3>{})
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, 3>, 5>(2, 3)
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, Eigen::Dynamic>, 5>(2, 3)
     * \endcode
     */
#ifdef __cpp_concepts
    template<vector_space_descriptor...Ds> requires 
      static_constant<MyConstant, ConstantType::static_constant> and 
      compatible_with_vector_space_descriptors<PatternMatrix, Ds...>
#else
    template<typename...Ds, std::enable_if_t<
      (vector_space_descriptor<Ds> and ...) and 
      static_constant<MyConstant, ConstantType::static_constant> and 
      compatible_with_vector_space_descriptors<PatternMatrix, Ds...>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(Ds&&...ds)
      : descriptor_collection {make_descriptors_collection(std::forward_as_tuple(std::forward<Ds>(ds)...))} {}

  private:

    template<std::size_t N = 0>
    constexpr auto make_dynamic_dimensions_tuple()
    {
      if constexpr (N < index_count_v<PatternMatrix>)
        return std::tuple_cat(std::forward_as_tuple(std::get<N>(descriptor_collection)), make_dynamic_dimensions_tuple<N + 1>());
      else
        return std::tuple {};
    }


    template<std::size_t N = 0, typename D, typename...Ds>
    constexpr auto make_dynamic_dimensions_tuple(D&& d, Ds&&...ds)
    {
      if constexpr (dynamic_vector_space_descriptor<std::tuple_element_t<N, DescriptorCollection>>)
        return std::tuple_cat(std::forward_as_tuple(std::forward<D>(d)),
          make_dynamic_dimensions_tuple<N + 1>(std::forward<Ds>(ds)...));
      else
        return std::tuple_cat(std::forward_as_tuple(std::get<N>(descriptor_collection)),
          make_dynamic_dimensions_tuple<N + 1>(std::forward<D>(d), std::forward<Ds>(ds)...));
    }

  public:

    /**
     * \brief Construct using only applicable \ref dynamic_vector_space_descriptor.
     * \tparam Ds A set of \ref dynamic_vector_space_descriptor corresponding to each of
     * class template parameter Ds that is dynamic, in order of Ds. This list should omit any \ref fixed_vector_space_descriptor.
     * \details If PatternMatrix has no dynamic dimensions, this is a default constructor.
     * The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, 2, dynamic_size>, 5>(std::integral_constant<int, 5>{}, 3) // Fixed rows and dynamic columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>>(5) // Fixed rows and columns.
     * ConstantAdapter<eigen_matrix_t<double, dynamic_size, 3>>(5, 2) // Dynamic rows and fixed columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, dynamic_size>, 5>(5, 3) // Fixed rows and dynamic columns.
     * \endcode
     */
#ifdef __cpp_concepts
    template<scalar_constant C, dynamic_vector_space_descriptor...Ds> requires
      vector_space_descriptor_tuple<DescriptorCollection> and 
      (dynamic_index_count_v<PatternMatrix> != dynamic_size) and 
      (sizeof...(Ds) == dynamic_index_count_v<PatternMatrix>) and 
      std::constructible_from<MyConstant, C&&>
#else
    template<typename C, typename...Ds, std::enable_if_t<
      scalar_constant<C> and (dynamic_vector_space_descriptor<Ds> and ...) and
      vector_space_descriptor_tuple<DescriptorCollection> and 
      (dynamic_index_count_v<PatternMatrix> != dynamic_size) and 
      (sizeof...(Ds) == dynamic_index_count_v<PatternMatrix>) and 
      std::is_constructible_v<MyConstant, C&&>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(C&& c, Ds&&...ds) 
      : my_constant {std::forward<C>(c)},
        descriptor_collection {make_dynamic_dimensions_tuple(std::forward<Ds>(ds)...)} {}


    /**
     * \overload
     * \brief Same as above, where the constant is known at compile time.
     * \details For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5>(2, 3) // Dynamic rows and columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, dynamic_size>, 5>(3) // Fixed rows and dynamic columns.
     * ConstantAdapter<eigen_matrix_t<double, dynamic_size, 3>, 5>(2) // Dynamic rows and fixed columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>, 5>() // Fixed rows and columns.
     * \endcode
     */
#ifdef __cpp_concepts
    template<dynamic_vector_space_descriptor...Ds> requires
      vector_space_descriptor_tuple<DescriptorCollection> and 
      (dynamic_index_count_v<PatternMatrix> != dynamic_size) and 
      (sizeof...(Ds) == dynamic_index_count_v<PatternMatrix>) and 
      scalar_constant<MyConstant, ConstantType::static_constant>
#else
    template<typename...Ds, std::enable_if_t<
      scalar_constant<C> and (dynamic_vector_space_descriptor<Ds> and ...) and
      vector_space_descriptor_tuple<DescriptorCollection> and 
      (dynamic_index_count_v<PatternMatrix> != dynamic_size) and 
      (sizeof...(Ds) == dynamic_index_count_v<PatternMatrix>) and 
      scalar_constant<MyConstant, ConstantType::static_constant>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(Ds&&...ds) 
      : descriptor_collection {make_dynamic_dimensions_tuple(std::forward<Ds>(ds)...)} {}


    /**
     * \brief Assign from a compatible \ref constant_matrix.
     */
#ifdef __cpp_concepts
    template<constant_matrix Arg> requires
      (not std::derived_from<Arg, ConstantAdapter>) and 
      vector_space_descriptors_may_match_with<Arg, PatternMatrix> and
      std::assignable_from<MyConstant, constant_coefficient<Arg>>
#else
    template<typename Arg, std::enable_if_t<constant_matrix<Arg> and
      (not std::is_base_of_v<ConstantAdapter, Arg>) and 
      vector_space_descriptors_may_match_with<Arg, PatternMatrix> and
      std::is_assignable_v<MyConstant, constant_coefficient<Arg>>, int> = 0>
#endif
    constexpr auto& operator=(const Arg& arg)
    {
      if constexpr (not vector_space_descriptors_match_with<Arg, PatternMatrix>) 
        if (not vector_space_descriptors_match(*this, arg)) throw std::invalid_argument {
          "Argument to ConstantAdapter assignment operator has non-matching vector space descriptors."};
      my_constant = constant_coefficient {arg};
      return *this;
    }


    /**
     * \brief Comparison operator
     */
#ifdef __cpp_concepts
    template<indexible Arg>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
    constexpr bool operator==(const Arg& arg) const
    {
      if constexpr (not vector_space_descriptors_may_match_with<Arg, PatternMatrix>)
        return false;
      else if constexpr (constant_matrix<Arg>)
        return get_scalar_constant_value(constant_coefficient{arg}) == get_scalar_constant_value(my_constant) and vector_space_descriptors_match(*this, arg);
      else
      {
        auto c = to_native_matrix<PatternMatrix>(*this);
        static_assert(not std::is_same_v<decltype(c), ConstantAdapter>,
          "interface::library_interface<PatternMatrix>::to_native_matrix(*this) must define an object within the library of Arg");
        return std::move(c) == arg;
      }
    }


#ifndef __cpp_impl_three_way_comparison
    /**
     * \overload
     */
#ifdef __cpp_concepts
    template<indexible Arg> (not constant_adapter<Arg>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and not constant_adapter<Arg>, int> = 0>
#endif
    friend constexpr bool operator==(const Arg& arg, const ConstantAdapter& c)
    {
      if constexpr (not vector_space_descriptors_may_match_with<Arg, PatternMatrix>)
        return false;
      else if constexpr (constant_matrix<Arg>)
        return get_scalar_constant_value(constant_coefficient{arg}) == get_scalar_constant_value(c.get_scalar_constant()) and vector_space_descriptors_match(arg, c);
      else
      {
        auto new_c = to_native_matrix<Arg>(c);
        static_assert(not std::is_same_v<decltype(new_c), ConstantAdapter>,
          "interface::library_interface<Arg>::to_native_matrix(c) must define an object within the library of Arg");
        return arg == std::move(new_c);
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
    constexpr bool operator!=(const Arg& arg) const { return not (*this == arg); }


#ifdef __cpp_concepts
    template<indexible Arg> (not std::same_as<Arg, ConstantAdapter>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and not std::is_same_v<Arg, ConstantAdapter>, int> = 0>
#endif
    friend constexpr bool operator!=(const Arg& arg, const ConstantAdapter& c) { return not (arg == c); }
#endif


    /**
     * \brief Access a component at a set of indices.
     * \return The element corresponding to the indices (always the constant).
     */
#ifdef __cpp_lib_ranges
    template<static_range_size<PatternMatrix> Indices> requires (not empty_object<PatternMatrix>)
    constexpr scalar_constant auto
#else
    template<typename Indices, std::enable_if_t<
      static_range_size<Indices, PatternMatrix> and (not empty_object<PatternMatrix>), int> = 0>
    constexpr auto
#endif
    operator[](const Indices& indices) const 
    {
      return get_scalar_constant_value(my_constant);
    }


    /**
     * \brief Access a component at a set of indices.
     * \return The element corresponding to the indices (always the constant).
     */
#ifdef __cpp_lib_ranges
    template<static_range_size<PatternMatrix> Indices> requires (not empty_object<PatternMatrix>)
    constexpr scalar_constant auto
#else
    template<typename Indices, std::enable_if_t<
      static_range_size<Indices, PatternMatrix> and (not empty_object<PatternMatrix>), int> = 0>
    constexpr auto
#endif
    operator()(const Indices& indices) const 
    {
      return get_scalar_constant_value(my_constant);
    }


    /**
     * \brief Get the \ref scalar_constant associated with this object.
     */
#ifdef __cpp_concepts
    constexpr scalar_constant auto
#else
    constexpr auto
#endif
    get_scalar_constant() const
    {
      return my_constant;
    }


    friend auto operator-(const ConstantAdapter& arg)
    {
      if constexpr (zero<ConstantAdapter>) return arg;
      else
      {
        values::scalar_constant_operation op {std::negate<>{}, constant_coefficient{arg}};
        return make_constant(arg, op);
      }
    }


#ifdef __cpp_concepts
    friend auto operator*(const ConstantAdapter& arg, scalar_constant auto s)
#else
    template<typename S, std::enable_if_t<scalar_constant<S>, int> = 0>
    friend auto operator*(const ConstantAdapter& arg, S s)
#endif
    {
      if constexpr (zero<decltype(arg)>) return arg;
      else
      {
        return make_constant(arg, constant_coefficient{arg} * s);
      }
    }


#ifdef __cpp_concepts
    friend auto operator*(scalar_constant auto s, const ConstantAdapter& arg)
#else
    template<typename S, std::enable_if_t<scalar_constant<S>, int> = 0>
    friend auto operator*(S s, const ConstantAdapter& arg)
#endif
    {
      if constexpr (zero<decltype(arg)>) return arg;
      else
      {
        return make_constant(arg, s * constant_coefficient{arg});
      }
    }


#ifdef __cpp_concepts
    friend auto operator/(const ConstantAdapter& arg, scalar_constant auto s)
#else
    template<typename S, std::enable_if_t<scalar_constant<S>, int> = 0>
    friend auto operator/(const ConstantAdapter& arg, S s)
#endif
    {
      return make_constant(arg, constant_coefficient{arg} / s);
    }

  protected:

    MyConstant my_constant;

    DescriptorCollection descriptor_collection;

    friend struct interface::indexible_object_traits<ConstantAdapter>;
    friend struct interface::library_interface<ConstantAdapter>;

  };


  // ------------------ //
  //  Deduction guides  //
  // ------------------ //

#ifdef __cpp_concepts
  template<scalar_constant C, indexible Arg>
#else
  template<typename C, typename Arg, std::enable_if_t<scalar_constant<C> and indexible<Arg>, int> = 0>
#endif
  ConstantAdapter(const C&, const Arg&) -> ConstantAdapter<Arg, C>;


#ifdef __cpp_concepts
  template<constant_matrix Arg> requires (not constant_adapter<Arg>)
#else
  template<typename Arg, std::enable_if_t<constant_matrix<Arg> and (not constant_adapter<Arg>), int> = 0>
#endif
  ConstantAdapter(const Arg&) -> ConstantAdapter<Arg, constant_coefficient<Arg>>;


  // ------------ //
  //  Interfaces  //
  // ------------ //

  namespace interface
  {
    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct indexible_object_traits<ConstantAdapter<PatternMatrix, Scalar, constant...>>
    {
    private:

      using XprType = ConstantAdapter<PatternMatrix, Scalar, constant...>;

    public:

      using scalar_type = typename XprType::MyScalarType;
      using MyDims = typename XprType::MyDimensions_t;


      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg)
      {
        if constexpr (index_count_v<PatternMatrix> == dynamic_size)
          return std::forward<Arg>(arg).descriptor_collection.size();
        else
          return std::tuple_size<MyDims>{};
      }


      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(Arg&& arg, const N& n)
      {
        if constexpr (index_count_v<PatternMatrix> == dynamic_size)
        {
          return std::forward<Arg>(arg).descriptor_collection[static_cast<typename MyDims::size_type>(n)];
        }
        else if constexpr (static_index_value<N>)
        {
          if constexpr (N::value >= index_count_v<PatternMatrix>) return Dimensions<1>{};
          else return std::get<N::value>(std::forward<Arg>(arg).descriptor_collection);
        }
        else if (n >= std::tuple_size_v<MyDims>)
        {
          return 1_uz;
        }
        else
        {
          return std::apply(
            [](auto&&...ds){ return std::array<std::size_t, std::tuple_size_v<MyDims>> {std::forward<decltype(ds)>(ds)...}; },
            std::forward<Arg>(arg).descriptor_collection)[n];
        }
      }


      // No nested_object defined


      template<typename Arg>
      static constexpr auto get_constant(const Arg& arg) { return arg.get_scalar_constant(); }


      // No get_constant_diagonal defined


      template<Qualification b>
      static constexpr bool one_dimensional = OpenKalman::one_dimensional<PatternMatrix, b>;


      // No is_square, is_triangular, is_triangular_adapter, or is_hermitian, or hermitian_adapter_type defined


      static constexpr bool is_writable = false;


      // No raw_data, layout, or strides defined.

    };


    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct library_interface<ConstantAdapter<PatternMatrix, Scalar, constant...>>
    {
      template<typename Derived>
      using LibraryBase = internal::library_base_t<Derived, PatternMatrix>;


      template<typename Arg, typename Indices>
      static constexpr auto
      get_component(Arg&& arg, const Indices&) { return std::forward<Arg>(arg).get_scalar_constant(); }


      // No set_component defined  because ConstantAdapter is not writable.


      template<typename Arg>
      static decltype(auto)
      to_native_matrix(Arg&& arg)
      {
        return OpenKalman::to_native_matrix<PatternMatrix>(std::forward<Arg>(arg));
      }


      template<Layout layout, typename S, typename...D>
      static auto
      make_default(D&&...d)
      {
        return make_dense_object<PatternMatrix, layout, S>(std::forward<D>(d)...);
      }


      // fill_components not necessary because T is not a dense writable matrix.


      template<typename C, typename...D>
      static constexpr auto
      make_constant(C&& c, D&&...d)
      {
        return OpenKalman::make_constant<PatternMatrix>(std::forward<C>(c), std::forward<D>(d)...);
      }


      template<typename S, typename...D>
      static constexpr auto
      make_identity_matrix(D&&...d)
      {
        return make_identity_matrix_like<PatternMatrix, S>(std::forward<D>(d)...);
      }


      // no get_slice
      // no set_slice
      // no set_triangle
      // no to_diagonal
      // no diagonal_of


      template<typename...Ds, typename Arg>
      static decltype(auto)
      replicate(const std::tuple<Ds...>& tup, Arg&& arg)
      {
        return library_interface<PatternMatrix>::replicate(tup, std::forward<Arg>(arg));
      }


      template<typename...Ds, typename Op, typename...Args>
      static constexpr decltype(auto)
      n_ary_operation(const std::tuple<Ds...>& d_tup, Op&& op, Args&&...args)
      {
        return library_interface<PatternMatrix>::n_ary_operation(d_tup, std::forward<Op>(op), std::forward<Args>(args)...);
      }


      template<std::size_t...indices, typename BinaryFunction, typename Arg>
      static constexpr decltype(auto)
      reduce(BinaryFunction&& b, Arg&& arg)
      {
        return library_interface<PatternMatrix>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
      }


      // no to_euclidean
      // no from_euclidean
      // no wrap_angles

      // conjugate is not necessary because it is handled by the general conjugate function.
      // transpose is not necessary because it is handled by the general transpose function.
      // adjoint is not necessary because it is handled by the general adjoint function.
      // determinant is not necessary because it is handled by the general determinant function.


      template<typename A, typename B>
      static constexpr auto sum(A&& a, B&& b)
      {
        return library_interface<PatternMatrix>::sum(std::forward<A>(a), std::forward<B>(b));
      }


      template<typename A, typename B>
      static constexpr auto contract(A&& a, B&& b)
      {
        return library_interface<PatternMatrix>::contract(std::forward<A>(a), std::forward<B>(b));
      }


      // contract_in_place is not necessary because the argument will not be writable.

      // cholesky_factor is not necessary because it is handled by the general cholesky_factor function.


      template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
      static decltype(auto) rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
      {
        using Trait = interface::library_interface<PatternMatrix>;
        return Trait::template rank_update_hermitian<significant_triangle>(std::forward<A>(a), std::forward<U>(u), alpha);
      }


      // rank_update_triangular is not necessary because it is handled by the general rank_update_triangular function.

      // solve is not necessary because it is handled by the general solve function.

      // LQ_decomposition is not necessary because it is handled by the general LQ_decomposition function.

      // QR_decomposition is not necessary because it is handled by the general QR_decomposition function.

    };

  } // namespace interface


} // namespace OpenKalman


#endif //OPENKALMAN_CONSTANTADAPTER_HPP
